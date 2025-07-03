import copy
from utils_lambda import *
from utils import convert_models_to_fp32
from torch.cuda import amp

from SO_Loss import pLoss_all_fidelity
from SO_Graph import graph_SA_ffso, graph_SO_FFSC

def create_task_flags_CLIP(task):
    semantic_tasks_3lvls = {'binary': 1, 'global': 1, 'local': 1}
    tasks = {}
    if task != 'all':
        tasks[task] = semantic_tasks_3lvls[task]
    else:
        tasks = semantic_tasks_3lvls
    return tasks

def get_weight_str(weight, tasks):
    """
    Record task weighting.
    """
    weight_str = 'Task Weighting | '
    for i, task_id in enumerate(tasks):
        weight_str += '{} {:.04f} '.format(task_id.title(), weight[i])
    return weight_str

def do_batch3_relative_similarity(model, x, joint_texts_prompts):
    batch_size = x.size(0)
    l1_texts = joint_texts_prompts[0][0]
    l2_texts = joint_texts_prompts[1][0]
    l3_texts = joint_texts_prompts[2][0]

    logits_per_image_1, _ = model.forward(x, l1_texts)
    logits_per_image_2, _ = model.forward(x, l2_texts)
    logits_per_image_3, _ = model.forward(x, l3_texts)

    logits_per_image_1 = logits_per_image_1.view(batch_size, -1)
    logits_per_image_2 = logits_per_image_2.view(batch_size, -1)
    logits_per_image_3 = logits_per_image_3.view(batch_size, -1)

    logits_per_image = torch.cat((logits_per_image_1, logits_per_image_2, logits_per_image_3),
                                 dim=1)
    return logits_per_image

class AutoLambda_SO:
    def __init__(self, model, device, train_tasks, pri_tasks, weight_init=0.1, dataset='so-ff'):
        self.model = model
        self.model_ = copy.deepcopy(model)
        convert_models_to_fp32(self.model)
        convert_models_to_fp32(self.model_)
        self.meta_weights = torch.tensor([weight_init] * len(train_tasks), requires_grad=True, device=device)
        self.train_tasks = train_tasks
        self.pri_tasks = pri_tasks

        if dataset == 'so-ff':
            self.criterion = pLoss_all_fidelity(graph_SA_ffso(), dataset)
        elif dataset == 'ffsc':
            self.criterion = pLoss_all_fidelity(hexG=graph_SO_FFSC(), dataset='ffsc')
        else:
            raise NotImplementedError
        print(dataset)

    def virtual_step(self, train_x, train_y, joint_texts, alpha, model_optim):
        """
        Compute unrolled network theta' (virtual step)
        """
        with amp.autocast(enabled=True):
            logits_joint = do_batch3_relative_similarity(self.model, train_x, joint_texts)

            _, train_loss = self.model_fit(logits_joint, train_y)

            loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])

        # compute gradient
        gradients = torch.autograd.grad(loss, self.model.parameters())

        # do virtual step (update gradient): theta' = theta - alpha * sum_i lambda_i * L_i(f_theta(x_i), y_i)
        with torch.no_grad():
            for weight, weight_, grad in zip(self.model.parameters(), self.model_.parameters(), gradients):
                if 'momentum' in model_optim.param_groups[0].keys():  # used in SGD with momentum
                    m = model_optim.state[weight].get('momentum_buffer', 0.) * model_optim.param_groups[0]['momentum']
                else:
                    m = 0
                weight_.copy_(weight - alpha * (m + grad + model_optim.param_groups[0]['weight_decay'] * weight))

    def unrolled_backward(self, train_x, train_y, joint_texts,
                          val_x, val_y,
                          alpha, alpha_lambda, model_optim):
        """
        Compute un-rolled loss and backward its gradients
        """

        # do virtual step (calc theta`)
        self.virtual_step(train_x, train_y, joint_texts, alpha, model_optim)

        # define weighting for primary tasks (with binary weights)
        pri_weights = []
        for t in self.train_tasks:
            if t in self.pri_tasks:
                pri_weights += [1.0]
            else:
                pri_weights += [0.0]

        with amp.autocast(enabled=True):
            logits_joint = do_batch3_relative_similarity(self.model_, val_x, joint_texts)
            _, val_loss = self.model_fit(logits_joint, val_y)

            loss = sum([w * val_loss[i] for i, w in enumerate(pri_weights)])

        # compute hessian via finite difference approximation
        model_weights_ = tuple(self.model_.parameters())

        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)

        hessian = self.compute_hessian(d_model, train_x, train_y, joint_texts)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip([self.meta_weights], hessian):
                mw.grad = - alpha_lambda * h

    def compute_hessian(self, d_model, train_x, train_y, joint_texts):
        norm = torch.cat([w.view(-1) for w in d_model]).norm() + 1e-6
        eps = 0.01 / norm

        # \theta+ = \theta + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d


        logits_joint = do_batch3_relative_similarity(self.model, train_x, joint_texts)
        _, train_loss = self.model_fit(logits_joint, train_y)

        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_p = torch.autograd.grad(loss, self.meta_weights)

        # \theta- = \theta - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p -= 2 * eps * d


        logits_joint = do_batch3_relative_similarity(self.model, train_x, joint_texts)
        _, train_loss = self.model_fit(logits_joint, train_y)

        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_n = torch.autograd.grad(loss, self.meta_weights)

        # recover theta
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        hessian = [(p - n) / (2. * eps) for p, n in zip(d_weight_p, d_weight_n)]
        return hessian

    def model_fit_(self, pred, targets):
        """
        define task specific losses
        """
        loss = [compute_loss(pred[i], targets[task_id], task_id) for i, task_id in enumerate(self.train_tasks)]
        return loss

    def model_fit(self, logits_joint, targets):
        try:
            total_loss, pMargin = self.criterion(logits_joint, targets, auto_mode=True)
        except:
            total_loss, pMargin = self.criterion(logits_joint, targets.float(), auto_mode=True)
        # for 3-level tasks
        loss_1 = total_loss[0]
        loss_2 = total_loss[1]
        loss_3 = total_loss[2]
        all_loss = [loss_1, loss_2, loss_3]

        return total_loss, all_loss