# import sys
# sys.path.append('/root/Meta/')
import math
import traceback
import copy

from data.load_dataset import CustomerDataLoader
from lib.utils.training_stats import TrainingStats
from lib.utils.evaluate_depth_error import validate_err
from lib.models.metric_depth_model import *
from lib.models.image_transfer import resize_image
from lib.core.config import cfg, merge_cfg_from_file, print_configs
from lib.utils.net_tools import save_ckpt, load_ckpt, save_ckpt_epoch
from lib.utils.logging import setup_logging, SmoothedValue
from tools.parse_arg_train import TrainOptions
from tools.parse_arg_val import ValOptions

logger = setup_logging(__name__)

def val(val_dataloader, model):
    smoothed_absRel = SmoothedValue(len(val_dataloader))
    smoothed_criteria = {'err_absRel': smoothed_absRel}
    val_num = 0
    for i, data in enumerate(val_dataloader):
        val_num += 1
        if val_num >= 500:
            break
        invalid_side = data['invalid_side'][0]
        out = model.module.inference(data)
        pred_depth = torch.squeeze(out['b_fake'])
        pred_depth = pred_depth[invalid_side[0]:pred_depth.size(0) - invalid_side[1], :]
        pred_depth = pred_depth / data['ratio'].cuda()
        pred_depth = resize_image(pred_depth, torch.squeeze(data['B_raw']).shape)
        smoothed_criteria = validate_err(pred_depth, data['B_raw'], smoothed_criteria, (45, 471, 41, 601))
    return {'abs_rel': smoothed_criteria['err_absRel'].GetGlobalAverageValue()}

if __name__ == '__main__':
    train_opt = TrainOptions()
    train_args = train_opt.parse()
    val_opt = ValOptions()
    val_args = val_opt.parse()
    val_args.batchsize = 1
    val_args.thread = 0
    merge_cfg_from_file(train_args)
    cfg.TRAIN.OUTPUT_DIR = cfg.TRAIN.OUTPUT_DIR + cfg.TRAIN.NAME
    cfg.TRAIN.LOG_DIR = cfg.TRAIN.OUTPUT_DIR

    train_dataloader = CustomerDataLoader(train_args)
    train_datasize = len(train_dataloader)
    gpu_num = torch.cuda.device_count()
    val_dataloader = CustomerDataLoader(val_args)
    val_datasize = len(val_dataloader)

    if train_args.use_tfboard:
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(cfg.TRAIN.LOG_DIR)
    training_stats = TrainingStats(train_args, cfg.TRAIN.LOG_INTERVAL, tblogger if train_args.use_tfboard else None)
    total_iters = math.ceil(train_datasize / train_args.batchsize) * train_args.epoch
    cfg.TRAIN.MAX_ITER = total_iters
    cfg.TRAIN.GPU_NUM = gpu_num

    MetaModel = MetricDepthModel()
    if gpu_num != -1:
        logger.info('{:>15}: {:<30}'.format('GPU_num', gpu_num))
        logger.info('{:>15}: {:<30}'.format('train_data_size', train_datasize))
        logger.info('{:>15}: {:<30}'.format('val_data_size', val_datasize))
        logger.info('{:>15}: {:<30}'.format('total_iterations', total_iters))
        MetaModel.cuda()

    MetaOptimizer = ModelOptimizer(MetaModel, parts=['decoder'], type='outer')

    Optimizer = ModelOptimizer(MetaModel, parts=[], type='all')

    lr_optim_lambda = lambda iter: (1.0 - iter / (float(total_iters))) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(MetaOptimizer.optimizer, lr_lambda=lr_optim_lambda)

    loss_func = ModelLoss()

    val_err = [{'abs_rel': 0}]
    ignore_step = -1
    if train_args.load_ckpt:
        load_ckpt(train_args, MetaModel, MetaOptimizer.optimizer, scheduler, val_err)
        ignore_step = train_args.start_step - train_args.start_epoch * math.ceil(train_datasize / train_args.batchsize)
    if gpu_num != -1:
        MetaModel = torch.nn.DataParallel(MetaModel)

    try:
        for epoch in range(train_args.start_epoch, train_args.epoch):
            MetaModel.train()
            epoch_steps = math.ceil(len(train_dataloader) / cfg.TRAIN.BATCHSIZE)
            base_steps = epoch_steps * epoch + ignore_step if ignore_step != -1 else epoch_steps * epoch

            for i, data in enumerate(train_dataloader):
                scheduler.step()
                training_stats.IterTic()

                model = copy.deepcopy(MetaModel)  # clone model
                optimizer = ModelOptimizer(model, parts=['decoder'], type='inner')

                out = model(data)
                losses = loss_func.criterion(out['b_fake_softmax'], out['b_fake_logit'], data, region='in_range')
                optimizer.optim(losses)

                out = model(data)
                losses = loss_func.criterion(out['b_fake_softmax'], out['b_fake_logit'], data, region='out_range')
                optimizer.backward(losses)
                MetaOptimizer.meta_update(MetaModel, model)

                out = model(data)
                losses = loss_func.criterion(out['b_fake_softmax'], out['b_fake_logit'], data)
                Optimizer.optim(losses)

                step = base_steps + i + 1
                training_stats.UpdateIterStats(losses)
                training_stats.IterToc()
                training_stats.LogIterStats(step, epoch, MetaOptimizer.optimizer, val_err[0])

                if step % cfg.TRAIN.VAL_STEP == 0 and step != 0 and val_dataloader is not None:
                    print(f"val for epoch {epoch}, step {step}")
                    model.eval()
                    val_err[0] = val(val_dataloader, model)
                    model.train()

                if step % cfg.TRAIN.SNAPSHOT_ITERS == 0 and step != 0:
                    save_ckpt(train_args, step, epoch, MetaModel, MetaOptimizer.optimizer, scheduler, val_err[0])
            ignore_step = -1

    except (RuntimeError, KeyboardInterrupt):
        logger.info('Save ckpt on exception ...')
        stack_trace = traceback.format_exc()
        print(stack_trace)

    finally:
        if train_args.use_tfboard:
            tblogger.close()
