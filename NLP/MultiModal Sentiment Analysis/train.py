import torch
import torch.nn as nn
import time
import numpy as np
import os
from utils.pred_func import *
from sklearn.metrics import classification_report


def train(net, train_loader, eval_loader, args):
    logfile = open(
        args.output + "/" + args.name + '/log_run.txt',
        'w'
    )
    logfile.write(str(args)+'\n')
    # exit()

    loss_sum = 0
    best_eval_accuracy = 0
    early_stop = 0
    decay_count = 0

    # Load the optimizer paramters
    optim = torch.optim.Adam(net.parameters(), lr=args.lr_base)

    loss_fn = args.loss_fn
    eval_accuracies = []
    for epoch in range(0, args.max_epoch):

        time_start = time.time()

        for step, (id, x, y, ans,) in enumerate(train_loader):
            # id:元组  x：tensor 80*60  y:tensor 80*60*80  sentiment:ans:80  emotion:ans:80*6

            loss_tmp = 0
            optim.zero_grad()

            x = x.cuda()
            y = y.cuda()

            ans = ans.cuda().long()
            # ans = ans.float()

            pred = net(x, y)

            loss = loss_fn(pred, ans)
            loss.backward()
            loss_sum += loss.cpu().data.numpy()
            loss_tmp += loss.cpu().data.numpy()

            print("\r[Epoch %2d][Step %4d/%4d] Loss: %.4f, Lr: %.2e, %4d m "
                  "remaining" % (
                      epoch + 1,
                      step,
                      int(len(train_loader.dataset) / args.batch_size),
                      loss_tmp / args.batch_size,  # 每一批数据中每一步的平均值
                      *[group['lr'] for group in optim.param_groups],
                      ((time.time() - time_start) / (step + 1)) * ((len(train_loader.dataset) / args.batch_size) - step) / 60,
                  ), end='          ')

            # Gradient norm clipping
            if args.grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    args.grad_norm_clip
                )

            optim.step()

        time_end = time.time()
        elapse_time = time_end - time_start
        print('Finished in {}s'.format(int(elapse_time)))
        epoch_finish = epoch + 1

        # Logging
        logfile.write(
            'Epoch: ' + str(epoch_finish) +
            ', Loss: ' + str(loss_sum / len(train_loader.dataset)) +
            ', Lr: ' + str([group['lr'] for group in optim.param_groups]) + '\n' +
            'Elapsed time: ' + str(int(elapse_time)) +
            ', Speed(s/batch): ' + str(elapse_time / step) +
            '\n\n'
        )

        # Eval
        if epoch_finish >= args.eval_start:
            print('Evaluation...')
            accuracy, _, _, _ = evaluate(net, eval_loader, args)
            print('Accuracy :' + str(accuracy))
            eval_accuracies.append(accuracy)
            if accuracy > best_eval_accuracy:
                # Best
                state = {
                    'state_dict': net.state_dict(),
                    'optimizer': optim.state_dict(),
                    'args': args,
                }
                if accuracy > 42.5:
                    torch.save(
                        state,
                        args.output + "/" + args.name +
                        '/best_%d_%.3f.pkl' % (args.seed, accuracy)
                    )
                best_eval_accuracy = accuracy
                early_stop = 0

            elif decay_count < args.lr_decay_times:
                # Decay
                print('LR Decay...')
                decay_count += 1
                # net.load_state_dict(torch.load(args.output + "/" + args.name +
                #                                '/best_%d_%4.f.pkl' % (args.seed, accuracy))['state_dict'])

                # adjust_lr(optim, args.lr_decay)
                for group in optim.param_groups:
                    group['lr'] *= args.lr_decay

            else:
                # Early stop
                early_stop += 1
                if early_stop == args.early_stop:
                    logfile.write('Early stop reached' + '\n')
                    print('Early stop reached')
                    logfile.write('best_overall_acc :' + str(best_eval_accuracy) + '\n\n')
                    print('best_eval_acc :' + str(best_eval_accuracy) + '\n\n')
                    os.rename(args.output + "/" + args.name +
                              '/best' + str(args.seed) + '.pkl',
                              args.output + "/" + args.name +
                              '/best' + str(best_eval_accuracy) + "_" + str(args.seed) + '.pkl')
                    logfile.close()
                    return eval_accuracies

        loss_sum = 0


def evaluate(net, eval_loader, args):
    accuracy = []
    net.train(False)
    prd_all = []
    ans_all = []
    preds = {}
    for step, (ids, x, y, ans,) in enumerate(eval_loader):
        x = x.cuda()
        y = y.cuda()
        pred = net(x, y).cpu().data.numpy()
        if len(prd_all) == 0:
            prd_all = pred
        else:
            prd_all = np.vstack((prd_all, pred))

        if not eval_loader.dataset.private_set:
            ans = ans.cpu().data.numpy()
            if len(ans_all) == 0:
                ans_all = ans
            else:
                ans_all = np.concatenate((ans_all, ans), axis=0)

            accuracy += list(eval(args.pred_func)(pred) == ans)
        # Save preds
        for id, p in zip(ids, pred):
            preds[id] = p

    net.train(True)
    print(classification_report(ans_all, eval(args.pred_func)(prd_all)))
    return 100 * np.mean(np.array(accuracy)), preds, prd_all, ans_all
