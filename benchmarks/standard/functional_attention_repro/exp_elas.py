import os
import argparse
import numpy as np
import torch
import time
from tqdm import *
import  Functional_Attention_Irregular_Mesh_Shared
from testloss import TestLoss
from normalizer import UnitTransformer
from experiment import (
    seed_everything,
    generate_seed,
    TrainingTimer,
    get_memory_usage
)

parser = argparse.ArgumentParser('Training Transformer')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='FuncAttn_2D')
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=3, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--basis_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=1)
parser.add_argument('--save_name', type=str, default='elas_FuncAttn')
parser.add_argument('--data_path', type=str, default='./')
parser.add_argument('--seed', type=int, default=None, help='random seed (None for auto-generate)')

parser.add_argument('--one_cycle_pct_start', type=float, default=0.1)
parser.add_argument('--one_cycle_div_factor', type=float, default=25)
parser.add_argument('--one_cycle_final_div_factor', type=float, default=10000)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

ntrain = args.ntrain
ntest = 200
eval = args.eval
save_name = args.save_name


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    seed = args.seed if args.seed is not None else generate_seed()
    seed_everything(seed)

    PATH_Sigma = args.data_path + '/elasticity/Meshes/Random_UnitCell_sigma_10.npy'
    PATH_XY = args.data_path + '/elasticity/Meshes/Random_UnitCell_XY_10.npy'

    input_s = np.load(PATH_Sigma)
    input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0)
    input_xy = np.load(PATH_XY)
    input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)

    train_s = input_s[:ntrain]
    test_s = input_s[-ntest:]
    train_xy = input_xy[:ntrain]
    test_xy = input_xy[-ntest:]

    print(input_s.shape, input_xy.shape)

    y_normalizer = UnitTransformer(train_s)
    train_s = y_normalizer.encode(train_s)
    y_normalizer.cuda()

    print("Dataloading is over.")

    g = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_xy, train_xy, train_s),
        batch_size=args.batch_size,
        shuffle=True,
        generator=g
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_xy, test_xy, test_s),
        batch_size=args.batch_size,
        shuffle=False
    )

    model = Functional_Attention_Irregular_Mesh_Shared.Model(
        space_dim=2,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        dropout=args.dropout,
        n_head=args.n_heads,
        Time_Input=False,
        mlp_ratio=args.mlp_ratio,
        fun_dim=0,
        out_dim=1,
        basis_num=args.basis_num,
        ref=args.ref,
        unified_pos=args.unified_pos
    ).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(args)
    print(model)
    total_params = count_parameters(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=args.one_cycle_pct_start,
        div_factor=args.one_cycle_div_factor,
        final_div_factor=args.one_cycle_final_div_factor,
    )

    myloss = TestLoss(size_average=False)

    run_name = f"{args.save_name}_basis{args.basis_num}_seed{seed}"

    config = {
        'lr': args.lr,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'basis_num': args.basis_num,
        'seed': seed,
        'total_params': total_params,
    }

    if eval:
        print("model evaluation")
        model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"), strict=False)
        model.eval()

        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')

        rel_err = 0.0

        with torch.no_grad():
            for pos, fx, y in test_loader:
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                out = model(x, None).squeeze(-1)
                out = y_normalizer.decode(out)

                tl = myloss(out, y).item()
                rel_err += tl

        rel_err /= ntest
        print("rel_err:{}".format(rel_err))

    else:
        timer = TrainingTimer()
        best_rel_err = float('inf')

        for ep in range(args.epochs):
            timer.start_epoch()
            torch.cuda.reset_peak_memory_stats()

            model.train()
            train_loss = 0

            for pos, fx, y in train_loader:
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                optimizer.zero_grad()

                out = model(x, None).squeeze(-1)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

                loss = myloss(out, y)
                loss.backward()

                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                train_loss += loss.item()
                scheduler.step()

            epoch_time = timer.end_epoch()
            max_mem = get_memory_usage()

            train_loss /= ntrain

            model.eval()
            rel_err = 0.0

            with torch.no_grad():
                for pos, fx, y in test_loader:
                    x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                    out = model(x, None).squeeze(-1)
                    out = y_normalizer.decode(out)

                    tl = myloss(out, y).item()
                    rel_err += tl

            rel_err /= ntest

            if rel_err < best_rel_err:
                best_rel_err = rel_err
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '_best.pt'))

            print("Epoch {} , train_loss:{:.5f} , test_loss:{:.5f} , best:{:.5f}, epoch_time_sec:{}, max_memory_MB:{:.5f}".format(
                ep, train_loss, rel_err, best_rel_err, e))

            # print({
            #     'epoch': ep,
            #     'train/l2_loss': train_loss,
            #     'test/rel_err': rel_err,
            #     'test/best_rel_err': best_rel_err,
            #     'lr': optimizer.param_groups[0]['lr'],
            #     'epoch_time_sec': epoch_time,
            #     'max_memory_MB': max_mem
            # })

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save model')
                torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))

        print({
            'best_rel_err': best_rel_err,
            'final_rel_err': rel_err,
            'total_train_time_sec': timer.total_time,
            'avg_epoch_time_sec': timer.get_avg_epoch_time(args.epochs),
            'total_params': total_params,
        })

        print(f"\n{'='*50}")
        print(f"Training Complete!")
        print(f"Best Rel Error: {best_rel_err:.6f}")
        print(f"Total Params: {total_params}")
        print(f"{'='*50}\n")

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('save model')
        torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))


if __name__ == "__main__":
    main()