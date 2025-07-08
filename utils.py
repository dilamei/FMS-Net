def clip_gradient(optimizer, grad_clip):
    """Clips gradients computed during backpropagation."""
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    """Decay learning rate by a factor of decay_rate every decay_epoch epochs."""
    if epoch < decay_epoch:
        current_lr = init_lr
    else:
        decay_times = epoch // decay_epoch
        decay = decay_rate ** decay_times
        current_lr = init_lr * decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    print('Epoch: {}, decay_epoch: {}, Current_LR: {}'.format(epoch, decay_epoch, current_lr))