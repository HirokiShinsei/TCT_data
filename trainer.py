import math
import sys
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc

import sys
sys.path.append(".")
sys.path.append("..")
import tool.utils as utils
from tool.voc_eval import write_custom_voc_results_file, do_python_eval
from tool.voc_eval_new import custom_voc_eval
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy


def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False


def train_process(model, optimizer, lr_sche,
                  dataloaders, num_epochs,
                  use_tensorboard,
                  device,
                  # model save params
                  save_model_path,
                  record_iter,
                  # tensorboard
                  writer=None,
                  ReduceLROnPlateau=False):
    savefig_flag = True
    model.train()
    model.apply(freeze_bn)
    
    ############################################################## fcos #####################################################################
    # running_loss = 0.0
    # running_loss_class = 0.0
    # running_loss_box_reg = 0.0
    # running_loss_ctrness = 0.0
    # best_score = 0.0
    # best_stat_dict = copy.deepcopy(model.state_dict())
    # for epoch in range(num_epochs):
    #     lr_scheduler = None
    #     print("====Epoch {0}====".format(epoch))
    #     if epoch == 0:
    #         warmup_factor = 1. / 1000
    #         warmup_iters = min(1000, len(dataloaders['train']) - 1)
    #         lr_scheduler = utils.warmup_lr_scheduler(optimizer,
    #                                                 warmup_iters,
    #                                                 warmup_factor)
    #     for i, (images, targets) in enumerate(tqdm(dataloaders['train']), 0):

    #         images = list(image.to(device) for image in images)
    #         targets = [{k: v.to(device) for k, v in t.items()}
    #                   for t in targets]
    #         optimizer.zero_grad()
    #         loss_dict = model(images, targets)
    #         losses = sum(loss for loss in loss_dict.values())

    #         # reduce losses over all GPUs for logging purposes
    #         loss_dict_reduced = utils.reduce_dict(loss_dict)
    #         losses_reduced = sum(loss for loss in loss_dict_reduced.values())

    #         loss_value = losses_reduced.item()

    #         if not math.isfinite(loss_value):
    #             print("Loss is {}, stopping training".format(loss_value))
    #             print(loss_dict_reduced)
    #             sys.exit(1)

    #         optimizer.zero_grad()
    #         losses.backward()
    #         optimizer.step()
    #         if lr_scheduler is not None:
    #             lr_scheduler.step()
    #         losses_total = losses.item()
    #         # 分类损失
    #         loss_class = loss_dict['classification'].item()
    #         # 回归损失
    #         loss_box_reg = loss_dict['bbox_regression'].item()
    #         # ctrness损失
    #         loss_ctrness = loss_dict['bbox_ctrness'].item()

    #         # 学习率
    #         lr = optimizer.param_groups[0]['lr']
    #         # lr_small = optimizer.param_groups[0]["lr"]
    #         # lr_large = optimizer.param_groups[1]["lr"]
    #         running_loss += losses_total
    #         running_loss_class += loss_class
    #         running_loss_box_reg += loss_box_reg
    #         running_loss_ctrness += loss_ctrness

    #         if (i+1) % record_iter == 0:
    #             print('''Epoch{0} loss:{1:.4f}
    #                      loss_class:{2:.4f} loss_box_reg:{3:.4f}
    #                      loss_ctrness:{4:.4f} \n'''.format(
    #                       epoch,
    #                       losses_total, loss_class,
    #                       loss_box_reg, loss_ctrness
    #                   ))
    #             if use_tensorboard:
    #                 # 写入tensorboard
    #                 writer.add_scalar("Total loss",
    #                                  running_loss / record_iter,
    #                                  epoch * len(dataloaders['train']) + i)
    #                 writer.add_scalar("classification loss",
    #                                  running_loss_class / record_iter,
    #                                  epoch * len(dataloaders['train']) + i)
    #                 writer.add_scalar("reg loss",
    #                                  running_loss_box_reg / record_iter,
    #                                  epoch * len(dataloaders['train']) + i)
    #                 writer.add_scalar("ctrloss",
    #                                  running_loss_ctrness / record_iter,
    #                                  epoch * len(dataloaders['train']) + i)
    #                 writer.add_scalar("lr",
    #                                  lr,
    #                                  epoch * len(dataloaders['train']) + i)
    #                 running_loss = 0.0
    #                 running_loss_class = 0.0
    #                 running_loss_box_reg = 0.0
    #                 running_loss_ctrness = 0.0


    ############################################################## ssd retinanet ###############################################
    # running_loss = 0.0
    # running_loss_class = 0.0
    # running_loss_box_reg = 0.0
    # best_score = 0.0
    # best_stat_dict = copy.deepcopy(model.state_dict())
    # for epoch in range(num_epochs):
    #     lr_scheduler = None
    #     print("====Epoch {0}====".format(epoch))
    #     if epoch == 0:
    #         warmup_factor = 1. / 1000
    #         warmup_iters = min(1000, len(dataloaders['train']) - 1)
    #         lr_scheduler = utils.warmup_lr_scheduler(optimizer,
    #                                                 warmup_iters,
    #                                                 warmup_factor)
    #     for i, (images, targets) in enumerate(tqdm(dataloaders['train']), 0):

    #         images = list(image.to(device) for image in images)
    #         targets = [{k: v.to(device) for k, v in t.items()}
    #                   for t in targets]
    #         optimizer.zero_grad()
    #         # 得到损失值字典
    #         loss_dict = model(images, targets)
    #         losses = sum(loss for loss in loss_dict.values())

    #         # reduce losses over all GPUs for logging purposes
    #         loss_dict_reduced = utils.reduce_dict(loss_dict)
    #         losses_reduced = sum(loss for loss in loss_dict_reduced.values())

    #         loss_value = losses_reduced.item()

    #         if not math.isfinite(loss_value):
    #             import ipdb;ipdb.set_trace()
    #             print("Loss is {}, stopping training".format(loss_value))
    #             print(loss_dict_reduced)
    #             sys.exit(1)

    #         optimizer.zero_grad()
    #         losses.backward()

    #         # 梯度裁剪(因损失值在一个epoch后出现了nan)
    #         # 限制一下梯度大小防止崩掉
    #         max_norm = 1.0  # 设置梯度裁剪的阈值
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    #         optimizer.step()
    #         if lr_scheduler is not None:
    #             lr_scheduler.step()
    #         losses_total = losses.item()
    #         # 分类损失
    #         loss_class = loss_dict['classification'].item()
    #         # 回归损失
    #         loss_box_reg = loss_dict['bbox_regression'].item()

    #         # 学习率
    #         lr = optimizer.param_groups[0]['lr']
    #         # lr_small = optimizer.param_groups[0]["lr"]
    #         # lr_large = optimizer.param_groups[1]["lr"]
    #         running_loss += losses_total
    #         running_loss_class += loss_class
    #         running_loss_box_reg += loss_box_reg

    #         if (i+1) % record_iter == 0:
    #             print('''Epoch{0} loss:{1:.4f}
    #                      loss_class:{2:.4f} loss_box_reg:{3:.4f}\n'''.format(
    #                       epoch,
    #                       losses_total, loss_class, loss_box_reg
    #                   ))
    #             if use_tensorboard:
    #                 # 写入tensorboard
    #                 writer.add_scalar("Total loss",
    #                                  running_loss / record_iter,
    #                                  epoch * len(dataloaders['train']) + i)
    #                 writer.add_scalar("classification loss",
    #                                  running_loss_class / record_iter,
    #                                  epoch * len(dataloaders['train']) + i)
    #                 writer.add_scalar("reg loss",
    #                                  running_loss_box_reg / record_iter,
    #                                  epoch * len(dataloaders['train']) + i)
    #                 writer.add_scalar("lr",
    #                                  lr,
    #                                  epoch * len(dataloaders['train']) + i)
    #                 running_loss = 0.0
    #                 running_loss_class = 0.0
    #                 running_loss_box_reg = 0.0
    
    
    #################################################### 两阶段训练记录 faster cascade ###########################################
    # running_loss = 0.0
    # running_loss_classifier = 0.0
    # running_loss_box_reg = 0.0
    # running_loss_objectness = 0.0
    # running_loss_rpn_box_reg = 0.0
    # best_score = 0.0
    # best_stat_dict = copy.deepcopy(model.state_dict())
    # for epoch in range(num_epochs):
    #     lr_scheduler = None
    #     print("====Epoch {0}====".format(epoch))
    #     if epoch == 0:
    #         warmup_factor = 1. / 1000
    #         warmup_iters = min(1000, len(dataloaders['train']) - 1)
    #         lr_scheduler = utils.warmup_lr_scheduler(optimizer,
    #                                                 warmup_iters,
    #                                                 warmup_factor)
    #     for i, (images, targets) in enumerate(tqdm(dataloaders['train']), 0):

    #         images = list(image.to(device) for image in images)
    #         targets = [{k: v.to(device) for k, v in t.items()}
    #                   for t in targets]
    #         optimizer.zero_grad()
    #         # 得到损失值字典
    #         loss_dict = model(images, targets)
    #         losses = sum(loss for loss in loss_dict.values())

    #         # reduce losses over all GPUs for logging purposes
    #         loss_dict_reduced = utils.reduce_dict(loss_dict)
    #         losses_reduced = sum(loss for loss in loss_dict_reduced.values())

    #         loss_value = losses_reduced.item()

    #         if not math.isfinite(loss_value):
    #             print("Loss is {}, stopping training".format(loss_value))
    #             print(loss_dict_reduced)
    #             sys.exit(1)

    #         optimizer.zero_grad()
    #         losses.backward()
    #         optimizer.step()
    #         if lr_scheduler is not None:
    #             lr_scheduler.step()
    #         losses_total = losses.item()
    #         # roi分类损失
    #         loss_classifier = loss_dict['loss_classifier'].item()
    #         # roi回归损失
    #         loss_box_reg = loss_dict['loss_box_reg'].item()
    #         # rpn分类损失
    #         loss_objectness = loss_dict['loss_objectness'].item()
    #         # rpn回归损失
    #         loss_rpn_box_reg = loss_dict['loss_rpn_box_reg'].item()
    #         # 学习率
    #         lr = optimizer.param_groups[0]['lr']
    #         # lr_small = optimizer.param_groups[0]["lr"]
    #         # lr_large = optimizer.param_groups[1]["lr"]
    #         running_loss += losses_total
    #         running_loss_classifier += loss_classifier
    #         running_loss_box_reg += loss_box_reg
    #         running_loss_objectness += loss_objectness
    #         running_loss_rpn_box_reg += loss_rpn_box_reg

    #         if (i+1) % record_iter == 0:
    #             print('''Epoch{0} loss:{1:.4f}
    #                      loss_classifier:{2:.4f} loss_box_reg:{3:.4f}
    #                      loss_objectness:{4:.4f} loss_rpn_box_reg:{5:.4f}\n'''.format(
    #                       epoch,
    #                       losses_total, loss_classifier,
    #                       loss_box_reg, loss_objectness,
    #                       loss_rpn_box_reg
    #                   ))
    #             if use_tensorboard:
    #                 # 写入tensorboard
    #                 writer.add_scalar("Total loss",
    #                                  running_loss / record_iter,
    #                                  epoch * len(dataloaders['train']) + i)
    #                 writer.add_scalar("RoI classification loss",
    #                                  running_loss_classifier / record_iter,
    #                                  epoch * len(dataloaders['train']) + i)
    #                 writer.add_scalar("RoI reg loss",
    #                                  running_loss_box_reg / record_iter,
    #                                  epoch * len(dataloaders['train']) + i)
    #                 writer.add_scalar("RPN classification loss",
    #                                  running_loss_objectness / record_iter,
    #                                  epoch * len(dataloaders['train']) + i)
    #                 writer.add_scalar("RPN reg loss",
    #                                  running_loss_rpn_box_reg / record_iter,
    #                                  epoch * len(dataloaders['train']) + i)
    #                 writer.add_scalar("lr",
    #                                  lr,
    #                                  epoch * len(dataloaders['train']) + i)
    #                 running_loss = 0.0
    #                 running_loss_classifier = 0.0
    #                 running_loss_box_reg = 0.0
    #                 running_loss_objectness = 0.0
    #                 running_loss_rpn_box_reg = 0.0


    ################################################# sparse rcnn ############################################
    running_loss = 0.0
    running_cls_loss = 0.0
    running_l1_loss = 0.0
    running_iou_loss = 0.0

    best_score = 0.0
    best_stat_dict = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        lr_scheduler = None
        print("====Epoch {0}====".format(epoch))
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(dataloaders['train']) - 1)
            lr_scheduler = utils.warmup_lr_scheduler(optimizer,
                                                    warmup_iters,
                                                    warmup_factor)
        for i, (images, targets) in enumerate(tqdm(dataloaders['train']), 0):

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                      for t in targets]
            optimizer.zero_grad()
            # 得到损失值字典
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            losses_total = losses.item()
            # 分类损失
            cls_loss = loss_dict['cls_loss'].item()
            # 回归损失
            l1_loss = loss_dict['l1_loss'].item()
            # iou损失
            iou_loss = loss_dict['iou_loss'].item()

            # 学习率
            lr = optimizer.param_groups[0]['lr']
            # lr_small = optimizer.param_groups[0]["lr"]
            # lr_large = optimizer.param_groups[1]["lr"]
            running_loss += losses_total
            running_cls_loss += cls_loss
            running_l1_loss += l1_loss
            running_iou_loss += iou_loss


            if (i+1) % record_iter == 0:
                print('''Epoch{0} loss:{1:.4f}
                         classification loss:{2:.4f} reg loss:{3:.4f}
                         iou loss:{4:.4f} \n'''.format(
                          epoch,
                          losses_total, cls_loss,
                          l1_loss, iou_loss
                      ))
                if use_tensorboard:
                    # 写入tensorboard
                    writer.add_scalar("Total loss",
                                     running_loss / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("classification loss",
                                     running_cls_loss / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("reg loss",
                                     running_l1_loss / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("iou loss",
                                     running_iou_loss / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("lr",
                                     lr,
                                     epoch * len(dataloaders['train']) + i)
                    running_loss = 0.0
                    running_cls_loss = 0.0
                    running_l1_loss = 0.0
                    running_iou_loss = 0.0


        valap_dict,val_mAP,valmf1 = custom_voc_evaluate(
            model, dataloaders["val"], device=device,
            gt_csv_path="/home/stat-zx/TCT_FIFTH/val.csv",
            cls_csv_path="/home/stat-zx/TCTdet/results/val_cls.csv",
            loc_csv_path="/home/stat-zx/TCTdet/results/val_loc.csv"
        )

        print(f"Epoch: {epoch}, | val TCT AP :{valap_dict['1']:.4f}")
        print(f"Epoch: {epoch}, | val mAP: {val_mAP:.4f}")
        
        if not ReduceLROnPlateau:
            lr_sche.step()
        else:
            lr_sche.step(val_mAP)
        
        if val_mAP > best_score:
            best_score = val_mAP
            best_stat_dict = copy.deepcopy(model.state_dict())
            savefig_flag = True
        else:
            savefig_flag = False
        
        if use_tensorboard:
            writer.add_scalar(
                'Validation mAP',
                val_mAP,
                global_step=epoch
            )
        
        model.train()
        model.apply(freeze_bn)


    print("===============训练完成===============")
    print(f"Best Valid mAP: {best_score:.4f}")
    torch.save(best_stat_dict, save_model_path)

    print("===============开始测试===============")
    model.load_state_dict(best_stat_dict)

    testap_dict,test_mAP,testmf1 = custom_voc_evaluate(
        model, dataloaders["test"], device=device,
            gt_csv_path="/home/stat-zx/TCT_FIFTH/test.csv",
            cls_csv_path="/home/stat-zx/TCTdet/results/test_cls.csv",
            loc_csv_path="/home/stat-zx/TCTdet/results/test_loc.csv"
    )

    print(f"Test TCT AP :{testap_dict['1']:.4f}")
    print(f"Test mAP: {test_mAP:.4f}")
    print(f"Test F1-score: {testmf1:.4f}")
    if use_tensorboard:
        writer.close()


@torch.no_grad()
def custom_voc_evaluate(model, data_loader, device,
                        gt_csv_path,
                        cls_csv_path,
                        loc_csv_path,
                        savefig_flag=False):                   
    cpu_device = torch.device('cpu')
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test'
    
    preds = [] 
    locs = []
    for image, _ in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        outputs = model(image)

        ########################################################## other models #############################################
        # outputs = [{k: v.to(cpu_device) for k, v in t.items()}
        #           for t in outputs]

        # if len(outputs[-1]["boxes"]) == 0:
        #     # if no pred boxes, means that the image is negative
        #     preds.append(0)
        #     locs.append("")

        # else:
        #     # we keep those pred boxes whose score is more than 0.5
        #     new_output_index = torch.where(outputs[-1]["scores"] > 0.5)
        #     new_boxes = outputs[-1]["boxes"][new_output_index]
        #     new_scores = outputs[-1]["scores"][new_output_index]
        #     new_labels = outputs[-1]["labels"][new_output_index]
        
        ################################################################ sparse rcnn #########################################
        outputs = {k: v.to(cpu_device) for k, v in outputs.items()}
        if len(outputs['boxes']) == 0:
            preds.append(0)
            locs.append("")
        else:
            new_output_index = torch.where(outputs['scores'] > 0.5)
            new_boxes = outputs['boxes'][new_output_index]
            new_scores = outputs['scores'][new_output_index]
            new_labels = outputs['labels'][new_output_index]


            TCT_index = torch.where(new_labels == 1)
            if len(new_boxes) != 0 and len(TCT_index[0]) != 0:               
                preds.append(torch.max(new_scores[TCT_index]).tolist())
            else:
                preds.append(0)
            
            # used to save pred coords x1 y1 x2 y2
            # used to save pred box scores
            coords = [] 
            for i in range(len(new_boxes)):
                new_box = new_boxes[i].tolist()
                coords.append([new_box[0], new_box[1],
                               new_box[2], new_box[3]])
            coords_score = new_scores.tolist()
            coords_labels = new_labels.tolist()
            line = ""
            for i in range(len(new_boxes)):
                if i == len(new_boxes) - 1:
                    line += str(coords_labels[i]) + ' ' + str(coords_score[i]) + ' ' +  \
                            str(coords[i][0]) + ' ' + str(coords[i][1]) + ' ' + \
                            str(coords[i][2]) + ' ' + str(coords[i][3])
                else:
                    line += str(coords_labels[i]) + ' ' + str(coords_score[i]) + ' ' + \
                            str(coords[i][0]) + ' ' + str(coords[i][1]) + ' ' + \
                            str(coords[i][2]) + ' ' + str(coords[i][3]) + ';'

            locs.append(line)

    print("====write cls pred results to csv====")
    cls_res = pd.DataFrame(
        {"image_path": data_loader.dataset.image_list,
         "prediction": preds}
    )
    cls_res.to_csv(cls_csv_path, columns=["image_path", "prediction"], sep=',', index=None)
    
    print("====write loc pred results to csv====") 
    loc_res = pd.DataFrame(
        {"image_path": data_loader.dataset.image_list,
         "prediction": locs}
    )
    loc_res.to_csv(loc_csv_path, columns=["image_path", "prediction"], sep=',', index=None)

    gt_anno = pd.read_csv(gt_csv_path, na_filter=False)
    AP_dict, mAP, mf1 = custom_voc_eval(gt_csv_path, loc_csv_path)
    return AP_dict,mAP,mf1



