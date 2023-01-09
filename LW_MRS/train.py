import torch
import os
import torch.backends.cudnn as cudnn
import argparse
from torch.utils.data import DataLoader
from MRSNet import MRSNet, weights_init

from TrainingDataLoad_global import *

parser = argparse.ArgumentParser(description='MRSNet')

parser.add_argument('--batchSize', type=int, default=64, help='Training batch size')
parser.add_argument('--testBatchSize', type=int, default=64, help='Testing batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=1, help='Random seed to use. Default=1')
opt = parser.parse_args()

class proj1():
    def __init__(self, config, training_set, testing_set=None):
        super(proj1, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.training_set = training_set
        self.testing_set = testing_set
        self.layer_number = 12

    def build_model(self):
        self.model = MRSNet(self.layer_number).to(self.device)
        # self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        ## Guassian initalization
        # self.model.apply(weights_init)
        self.criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='sum')

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 100, 150, 200, 250], gamma=0.7)

    def save_checkpoint(self, epoch):
        model_out_path = "checkpoint_MRSNet_global/" + "model_epoch_{}.pth".format(epoch)
        state = {"epoch": epoch, "model": self.model}
        if not os.path.exists("checkpoint_MRSNet_global/"):
            os.makedirs("checkpoint_MRSNet_global/")

        torch.save(state, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()

        print('<==============================================================Train start==============================================================>')

        train_loss_pred = 0
        train_loss_sym = 0
        for i, data in enumerate(self.training_set):
            image_r, image_i, masks, label_r, label_i, linewidth = data
            image_r = image_r.to(self.device)
            image_i = image_i.to(self.device)
            label_r = label_r.to(self.device)
            label_i = label_i.to(self.device)
            masks = masks.to(self.device)

            # lw = lw.to(self.device)

            self.optimizer.zero_grad()

            pred_r, pred_i, floss_r, floss_i = self.model(image_r, image_i, masks, linewidth, device=self.device)

            loss1 = self.criterion(pred_r, label_r)
            loss2 = self.criterion(pred_i, label_i)

            loss_constraint = torch.mean(torch.pow(floss_r[0], 2))
            for k in range(self.layer_number - 1):
                loss_constraint += torch.mean(torch.pow(floss_r[k + 1], 2))
                loss_constraint += torch.mean(torch.pow(floss_i[k + 1], 2))

            loss = loss1 + loss2 + 5e-5 * loss_constraint

            train_loss_pred += (loss1 + loss2).item()
            train_loss_sym += loss_constraint.item()
            loss.backward()

            self.optimizer.step()
            # self.optimizer.zero_grad()

        print("    Pred Loss: {}".format(train_loss_pred))
        print("    Sym Loss: {}".format(train_loss_sym))
        print('    Learning Rate: {}'.format(self.optimizer.param_groups[0]['lr']))

    # def test(self):
    #     self.model.eval()
    #     print('<==============================================================Test start==============================================================>')
    #     train_loss_pred = 0
    #     train_loss_sym = 0
    #     with torch.no_grad():
    #         for i, data in enumerate(self.training_set):
    #
    #             image_r, image_i, masks, label_r, label_i = data
    #             image_r = image_r.to(self.device)
    #             image_i = image_i.to(self.device)
    #             label_r = label_r.to(self.device)
    #             label_i = label_i.to(self.device)
    #             masks = masks.to(self.device)
    #
    #             pred_r, pred_i, floss_r, floss_i = self.model(image_r, image_i, masks, self.device)
    #
    #             loss1 = self.criterion(pred_r, label_r)
    #             loss2 = self.criterion(pred_i, label_i)
    #
    #             loss_constraint = torch.mean(torch.pow(floss_r[0], 2))
    #             for k in range(self.layer_number - 1):
    #                 loss_constraint += torch.mean(torch.pow(floss_r[k + 1], 2))
    #                 loss_constraint += torch.mean(torch.pow(floss_i[k + 1], 2))
    #
    #             loss_pred = loss1 + loss2
    #
    #             train_loss_pred += loss_pred.item()
    #             train_loss_sym += loss_constraint.item()
    #
    #     print("   Pred Loss: {}".format(train_loss_pred))
    #     print("   Sym Loss: {}".format(train_loss_sym))

    def run(self):
        self.build_model()
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            # self.test()
            self.scheduler.step()
            if epoch % 20 == 0:
                self.save_checkpoint(epoch)

def main():
    opt = parser.parse_args()
    print(opt)

    print("===> Loading datasets")
    # Load train set  "C:/Users/s4548361/Desktop/Train_data_350_32or16/train_dataset_350_32/"

    train_path = "C:/Users/s4548361/Desktop/LW_MRS_NOISE_FREE/train_dataset_350_noise_free/"
    train_dataset = DataSet(train_path)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=opt.batchSize, drop_last=True)
    # # Load test set
    # test_path = 'test_dataset_350/'
    # test_dataset = Dataset(test_path)
    # test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=opt.testBatchSize, drop_last=True)
    model = proj1(opt, train_dataloader)
    model.run()


if __name__ == "__main__":
    main()