import os
import torch
import pickle
from model import RDFNet as net
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
import torch.optim.lr_scheduler

from loss import TotalLoss

def train_net(args):
    # load the model
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    model = net.RDFNet()

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    args.savedir = args.savedir + '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)


    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(valid=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    criteria = TotalLoss()

    start_epoch = 0
    lr = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # Adjust keys in state_dict
            state_dict = checkpoint['state_dict']
            new_state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith('module.')}
        
            start_epoch = checkpoint['epoch']
            # model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(new_state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # for epoch in range(start_epoch, args.max_epochs):
    #     print(f"Starting epoch {epoch}")
    #     model_file_name = args.savedir + os.sep + 'model_{}.pth'.format(epoch)
    #     poly_lr_scheduler(args, optimizer, epoch)
    #     for param_group in optimizer.param_groups:
    #         lr = param_group['lr']
    #         #   lr=1.8e-06
    #     print("Learning rate: " +  str(lr))

    #     # train for one epoch
    #     model.train()
    #     train( args, trainLoader, model, criteria, optimizer, epoch)
    #     model.eval()
    #     # validation
    #     # val(valLoader, model)
    #     # ----------------------------------------validation and print result-------------------------#
    #     da_segment_results,ll_segment_results = val(valLoader, model)

    #     msg =  'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
    #                     'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})'.format(
    #                         da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
    #                         ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2])
    #     print(msg)
    #     print("loss::")
    #     val
    #     torch.save(model.state_dict(), model_file_name)
        
    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'lr': lr
    #     }, args.savedir + 'checkpoint.pth.tar')
    
    for epoch in range(start_epoch, args.max_epochs):
        print(f"Starting epoch {epoch}")
        model_file_name = args.savedir + os.sep + 'model_{}.pth'.format(epoch)
        poly_lr_scheduler(args, optimizer, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " +  str(lr))

        # train for one epoch
        model.train()
        train(args, trainLoader, model, criteria, optimizer, epoch)
        model.eval()

        # validation
        da_segment_results, ll_segment_results, fps = val(valLoader, model)

        msg = ('Driving area Segment: Acc({da_seg_acc:.3f}) IOU ({da_seg_iou:.3f}) mIOU({da_seg_miou:.3f})\n'
            'Lane line Segment: Acc({ll_seg_acc:.3f}) IOU ({ll_seg_iou:.3f}) mIOU({ll_seg_miou:.3f})\n'
            'FPS: {fps:.2f}').format(
                da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],
                ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2],
                fps=fps)
        print(msg)

        print("loss::")
        torch.save(model.state_dict(), model_file_name)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr
        }, args.savedir + 'checkpoint.pth.tar')
        



if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=200, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--step_loss', type=int, default=40, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--savedir', default='./test_', help='directory to save the results')
    parser.add_argument('--resume', type=str, default='/media/usman/disk2/reslite7/test_/checkpoint.pth.tar', help='Use this flag to load last checkpoint for training')
    parser.add_argument('--pretrained', default='', help='Pretrained ESPNetv2 weights.')

    train_net(parser.parse_args())

