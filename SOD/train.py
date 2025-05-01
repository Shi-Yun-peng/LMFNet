import os, torch, random, pickle, time
import datetime
import data as datta
import models
import loss as ls
import transforms
from utils import SalEval, AverageMeterSmooth, Logger, plot_training_process
from torch.utils.data import DataLoader
from glob import glob
import math

params = {
    'size':256,
    'batch_size': 8,
    'max_epochs': 200,
    'data': 'DUTS_T',
     'scheduler':'exp',
    'model_name': 'LMFNet',
    'decay_rate': 0.98,
    'dataname' : 'DUTS_T',
    'save_dir': './RESULT',
    'print_flag': 400,
    'lr' : 0.001,

}

save_dir=params['save_dir']
data_name=params['dataname']
cached_data_file: str = f'./data/{data_name}/train/train256.p'
val_cached_data_file: str = f'./data/{data_name}/val/val256.p'
lr=params['lr']
resume=os.path.join(save_dir, 'checkpoint.pth')


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def val(val_loader, epoch):
    # switch to evaluation mode
    model.eval()
    print("val")
    salEvalVal = SalEval()
    total_batches = len(val_loader)
    for iter, (input, target) in enumerate(val_loader):

        input = input.to(device)
        target = target.to(device)
        input = torch.autograd.Variable(input)
        target = torch.autograd.Variable(target)

        start_time = time.time()

        output= model(input)
        torch.cuda.synchronize()
        val_times.update(time.time() - start_time)

        loss= los(output, target)

        val_losses.update(loss.item())



        salEvalVal.addBatch(output[:, 0, :, :], target)

        if iter % params['print_flag'] == 0:
            logger.info('Epoch [%d/%d] Iter [%d/%d] Time: %.3f loss: %.3f (avg: %.3f)' %
                        (epoch, params['max_epochs'], iter, total_batches, val_times.avg,
                        val_losses.val, val_losses.avg))

    F_beta, MAE = salEvalVal.getMetric()
    record['val']['F_beta'].append(F_beta)
    record['val']['MAE'].append(MAE)

    return F_beta, MAE



def train(train_loader, epoch, cur_iter=0):

    model.train()
    salEvalTrain = SalEval()

    total_batches = len(train_loader)

    end = time.time()

    for iter, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)

        input = torch.autograd.Variable(input)
        target = torch.autograd.Variable(target)

        start_time = time.time()
        # run the mdoel
        output= model(input)
        loss= los(output, target)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()


        train_losses.update(loss.item())
        record[0].append(train_losses.avg)
        train_batch_times.update(time.time() - start_time)             
        train_data_times.update(start_time - end)                      

        salEvalTrain.addBatch(output[:, 0, :, :], target)


        if iter % params['print_flag'] == 0:
            logger.info('Epoch [%d/%d] Iter [%d/%d] Batch time: %.3f Data time: %.3f ' \
                        'loss: %.3f  lr: %.1e' % \
                        (epoch, params['max_epochs'],
                         iter + cur_iter, total_batches + cur_iter, \
                         train_batch_times.avg,
                         train_data_times.avg, \
                         train_losses.val,  lr))

        end = time.time()

    F_beta, MAE = salEvalTrain.getMetric()
    record['train']['F_beta'].append(F_beta)
    record['train']['MAE'].append(MAE)

    return F_beta, MAE

if __name__ == '__main__':
    
    seed =34   
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    exec('from models import {} as net'.format(params['model_name']))
    model=net()
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    log_name = 'log_' + datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S') + '.txt'
    logger = Logger(os.path.join(save_dir, log_name))
    logger.info('Called with args:')
    
    for key, value in params.items():
        logger.info('{0:16} | {1}'.format(key, value))

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params}")
    img_ids = glob(os.path.join(f'./data/{data_name}/train/input', '*.jpg'))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    if not os.path.isfile(cached_data_file):
        data_loader = datta.LoadData(dataname=params['dataname'],dataclass='train',
                                     cached_data_file=os.path.join(f'./data/{data_name}/train/train256.p'), img_ids=img_ids)
        data= data_loader.process()


        if data is None:
            logger.info('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(f'./data/{data_name}/train/train256.p', 'rb'))

    mean = data['mean']
    std = data['std']

    val_ids = glob(os.path.join(f'./data/{data_name}/val/input', '*.jpg'))
    val_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_ids]
    if not os.path.isfile(val_cached_data_file):
        val_data_loader = datta.LoadData(dataname=params['dataname'],dataclass='val',
                                    cached_data_file=os.path.join(f'./data/{data_name}/val/val256.p'), img_ids=val_ids)
        val_data = val_data_loader.process()

        if val_data is None:
            logger.info('Error while pickling data. Please check.')
            exit(-1)
    else:
        val_data = pickle.load(open(f'./data/{data_name}/val/val256.p', 'rb'))
        
    val_mean = val_data['mean']
    val_std = val_data['std']
    logger.info('Data statistics:')
    logger.info('mean: [%.5f, %.5f, %.5f], std: [%.5f, %.5f, %.5f]' % (*data['mean'], *data['std']))


    los = ls.CrossEntropyLoss()
    los = los.to(device)
    train_losses = AverageMeterSmooth()
    train_batch_times = AverageMeterSmooth()
    train_data_times = AverageMeterSmooth()

    val_losses = AverageMeterSmooth()
    val_times = AverageMeterSmooth()
    valTransform = transforms.Compose([
        transforms.Normalize(mean=val_data['mean'], std=val_data['std']),
        transforms.Scale(params['size'],params['size'] ),
        transforms.VToTensor()
        ])

    
    val_set = datta.Dataset(dataname=params['dataname'],dataclass='val', img_ids=val_ids, transform=valTransform)
    val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=params['batch_size'], shuffle=False,
            num_workers=1, pin_memory=True
        )
    
    trainTransform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Normalize(mean=data['mean'], std=data['std']),
            transforms.RandomCropResize(10),
            transforms.Scale(params['size'],params['size']),
            transforms.RandomFlip(),
            transforms.TToTensor()
            ])
    train_set = datta.Dataset(dataname=params['dataname'],dataclass='train', img_ids=img_ids, transform=trainTransform)
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=params['batch_size'] , shuffle=True,
            num_workers=1, pin_memory=True, drop_last=True
            )
    
    record = {
        0: [],'lr': [],
        'val': {'F_beta': [], 'MAE': []},
        'train': {'F_beta': [], 'MAE': []}
    }
    bests = {'F_beta_tr': 0., 'F_beta_val': 0., 'MAE_tr': 1., 'MAE_val': 1.}
    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,  gamma=params['decay_rate'])
    logger.info('Optimizer Info:\n' + str(optimizer))

    start_epoch = 0                                                              
    if os.path.isfile(resume):
        logger.info('=> loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
        logger.info('=> loaded checkpoint {} (epoch {})'.format(resume, checkpoint['epoch']))
    else:
        logger.info('=> no checkpoint found at {}'.format(resume))


        
    for epoch in range(start_epoch, params['max_epochs']): 
        lr = optimizer.param_groups[0]['lr']
        record['lr'].append(lr)
        print(lr)

        F_beta_tr, MAE_tr= train(train_loader, epoch, 0)
        F_beta_val, MAE_val= val(val_loader, epoch)

        if F_beta_tr > bests['F_beta_tr']: bests['F_beta_tr'] = F_beta_tr
        if MAE_tr < bests['MAE_tr']: bests['MAE_tr'] = MAE_tr
        if F_beta_val > bests['F_beta_val']: bests['F_beta_val'] = F_beta_val
        
        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_F_beta': bests['F_beta_val'],
            'best_MAE': bests['MAE_val']
            }, os.path.join(save_dir, 'checkpoint.pth'))

        model_file_name = os.path.join(save_dir, 'model' + str(epoch + 1)+'--'+str(MAE_val)  + '.pth')
        torch.save(model.state_dict(), model_file_name)

        if MAE_val < bests['MAE_val']:bests['MAE_val'] = MAE_val    
        logger.info('Epoch %d: F_beta (tr) %.4f (Best: %.4f) MAE (tr) %.4f (Best: %.4f) ' \
                    'F_beta (val) %.4f (Best: %.4f) MAE (val) %.4f (Best: %.4f)' % \
                    (epoch, F_beta_tr, bests['F_beta_tr'], MAE_tr, bests['MAE_tr'], \
                     F_beta_val, bests['F_beta_val'], MAE_val, bests['MAE_val']))
        plot_training_process(record, save_dir, bests)
        
        scheduler.step()
    logger.close()

