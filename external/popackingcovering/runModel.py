import sys
import os
import numpy as np
import random
import pandas as pd
import math, time
import itertools
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import datetime
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import gurobipy as gp
import logging
import copy
from collections import defaultdict
import joblib
import sys
import gurobipy as gp
from gurobipy import GRB
import argparse
import sys
import external.popackingcovering.ip_model_whole as ip_model_wholeFile
from external.popackingcovering.ip_model_whole import IPOfunc

########################################################################################################################

itemNum = 10
featureNum = 4096
trainSize = 700
targetNum = 2
meanPriceValue = 0
meanWeightValue = 0
testi = 0
testTime = 3

########################################################################################################################


def actual_obj(valueTemp, cap, weightTemp, n_instance):
    obj_list = []
    for num in range(n_instance):
        weight = np.zeros(itemNum)
        value = np.zeros(itemNum)
        cnt = num * itemNum
        for i in range(itemNum):
            weight[i] = weightTemp[cnt]
            value[i] = valueTemp[cnt]
            cnt = cnt + 1
        weight = weight.tolist()
        value = value.tolist()

        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(itemNum, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
        m.setObjective(x.prod(value), GRB.MAXIMIZE)
        m.addConstr((x.prod(weight)) <= cap)
        #        for i in range(itemNum):
        #            m.addConstr((x.prod(weight[i])) <= cap)

        m.optimize()
        #        sol = []
        #        for i in range(itemNum):
        #            sol.append(x[i].x)
        #        print(sol)
        objective = m.objVal
        obj_list.append(objective)
    #        print(objective)

    return np.array(obj_list)

########################################################################################################################


def correction_single_obj(realPrice, predPrice, cap, realWeightTemp, predWeightTemp, penalty):
    realWeight = np.zeros(itemNum)
    predWeight = np.zeros(itemNum)
    realPriceNumpy = np.zeros(itemNum)

    feasible = True

    for i in range(itemNum):
        realWeight[i] = realWeightTemp[i]
        predWeight[i] = predWeightTemp[i]
        realPriceNumpy[i] = realPrice[i]

    if min(predWeight) >= 0:
        predWeight = predWeight.tolist()
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(itemNum, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
        m.setObjective(x.prod(predPrice), GRB.MAXIMIZE)
        m.addConstr((x.prod(predWeight)) <= cap)

        m.optimize()
        sol = []
        for i in range(itemNum):
            sol.append(x[i].x)
#        print("realPrice: ", realPrice)
#        print("predPrice: ", predPrice)
#        print(sol)
#        objective = m.objVal
        objective = 0
        for i in range(itemNum):
            objective = objective + sol[i] * realPrice[i]
#        print(m.objVal, objective)
#        print(objective)

        #correction
        tau = 1
        selectedTotalWeight = np.dot(realWeight, sol)
        if selectedTotalWeight > cap:
            tau = cap / selectedTotalWeight
            feasible = False
        objective = tau * objective - np.dot(np.multiply(realPriceNumpy, penalty), np.multiply(sol, 1-tau))
        sol = np.multiply(sol, tau)

    else:
        objective = 0
        feasible = False

#    print(sol)
#    print(objective)
#        print(np.dot(G, sol), objective)
#        print("")
    return objective, feasible

########################################################################################################################

# simply define a silu function
def silu(input):
    for i in range(itemNum):
        # if input[i][0] < 0:
        #     input[i][0] = 0
        input[i][0] = input[i][0] + ReLUValue
        # if input[i][1] < 0:
        #     input[i][1] = 0
        input[i][1] = input[i][1] + ReLUValue
    return input

########################################################################################################################

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class SiLU(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        return silu(input) # simply apply already implemented SiLU

########################################################################################################################


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

########################################################################################################################


def make_fc(num_layers, num_features, num_targets=targetNum,
            activation_fn = nn.ReLU,intermediate_size=512, regularizers = True):

    # initialize activation function
    activation_function = SiLU()

    net_layers = [nn.Linear(num_features, intermediate_size), activation_function]
    for hidden in range(num_layers-2):
        net_layers.append(nn.Linear(intermediate_size, intermediate_size))
        net_layers.append(activation_function)
    net_layers.append(nn.Linear(intermediate_size, num_targets))
    net_layers.append(activation_function)

    net_layers = [nn.Linear(num_features, num_targets), activation_function]

    return nn.Sequential(*net_layers)

########################################################################################################################


class MyCustomDataset():
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

    def __len__(self):
        return len(self.value)

    def __getitem__(self, idx):
        return self.feature[idx], self.value[idx]

########################################################################################################################


class Intopt:
    def __init__(self, c, h, A, b, penalty, n_features, num_layers=5, smoothing=False, thr=0.1, max_iter=None, method=1, mu0=None,
                 damping=0.5, target_size=targetNum, epochs=8, optimizer=optim.Adam,
                 batch_size=itemNum, **hyperparams):
        self.c = c
        self.h = h
        self.A = A
        self.b = b
        self.penalty = penalty
        self.target_size = target_size
        self.n_features = n_features
        self.damping = damping
        self.num_layers = num_layers

        self.smoothing = smoothing
        self.thr = thr
        self.max_iter = max_iter
        self.method = method
        self.mu0 = mu0

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.hyperparams = hyperparams
        self.epochs = epochs
        # print("embedding size {} n_features {}".format(embedding_size, n_features))

#        self.model = Net(n_features=n_features, target_size=target_size)
        self.model = make_fc(num_layers=self.num_layers,num_features=n_features)
        #self.model.apply(weight_init)
#        w1 = self.model[0].weight
#        print(w1)

        self.optimizer = optimizer(self.model.parameters(), **hyperparams)

    def fit(self, feature, value):
        logging.info("Intopt")
        train_df = MyCustomDataset(feature, value)

        criterion = nn.L1Loss(reduction='mean')  # nn.MSELoss(reduction='mean')
        grad_list = np.zeros(self.epochs)
        for e in range(self.epochs):
          total_loss = 0
#          for parameters in self.model.parameters():
#            print(parameters)
          if e < 0:
            #print('stage 1')
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size, shuffle=False)
            for feature, value in train_dl:
                self.optimizer.zero_grad()
                op = self.model(feature).squeeze()
#                print(feature, value, op)
#                print(feature.shape, value.shape, op.shape)
                # targetNum=1: torch.Size([10, 4096]) torch.Size([10]) torch.Size([10])
                # targetNum=2: torch.Size([10, 4096]) torch.Size([10, 2]) torch.Size([10, 2])
#                print(value, op)

                loss = criterion(op, value)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
          else:
#            if e > 4:
#                for param_group in self.optimizer.param_groups:
#                    param_group['lr'] = 1e-10
            #print('stage 2')
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size, shuffle=False)

            num = 0
            batchCnt = 0
            loss = Variable(torch.tensor(0.0, dtype=torch.double), requires_grad=True)
            for feature, value in train_dl:
                self.optimizer.zero_grad()
                op = self.model(feature).squeeze()
                while torch.min(op) <= 0 or torch.isnan(op).any() or torch.isinf(op).any():
                    self.optimizer.zero_grad()
#                    self.model.__init__(self.n_features, self.target_size)
                    self.model = make_fc(num_layers=self.num_layers,num_features=self.n_features)
                    op = self.model(feature).squeeze()

                price = np.zeros(itemNum)
                penaltyVector = np.zeros(itemNum)
                for i in range(itemNum):
                    price[i] = self.c[i+num*itemNum]
                    penaltyVector[i] = self.penalty[i+num*itemNum]
                    op[i] = op[i]

                c_torch = torch.from_numpy(price).float()
                h_torch = torch.from_numpy(self.h).float()
                A_torch = torch.from_numpy(self.A).float()
                b_torch = torch.from_numpy(self.b).float()
                penalty_torch = torch.from_numpy(penaltyVector).float()

                G_torch = torch.zeros((itemNum+1, itemNum))
                for i in range(itemNum):
                    G_torch[i][i] = 1
                G_torch[itemNum] = value[:, 1]

#                op_torch = torch.zeros((itemNum+1, itemNum))
#                for i in range(itemNum):
#                    op_torch[i][i] = 1
#                op_torch[itemNum] = op

#                print(G_torch)
#                print(op_torch)
                x = IPOfunc(A=A_torch, b=b_torch, h=h_torch, cTrue=-c_torch, GTrue=G_torch, penalty=penalty_torch, max_iter=self.max_iter, thr=self.thr, damping=self.damping,
                            smoothing=self.smoothing)(op)
                #print(c_torch.shape, G_torch.shape, x.shape)    # torch.Size([242]) torch.Size([43, 242]) torch.Size([242])
                x_org = x / ip_model_wholeFile.violateFactor
#                print(x, c_torch)
#                newLoss = (x * c_torch).sum() + torch.dot(torch.mul(c_torch, penalty), torch.mul(x, 1-1/ip_model_wholeFile.violateFactor))
                newLoss = (x_org * -c_torch).sum() + torch.dot(torch.mul(-c_torch, 1+penalty_torch), torch.mul(x_org, ip_model_wholeFile.violateFactor - 1))
#                print(newLoss)
#                newLoss = - (x * c_torch).sum()
#                loss = loss - (x * c_torch).sum()
                loss = loss + newLoss
                batchCnt = batchCnt + 1
#                print(loss)
#                loss = torch.dot(-c_torch, x)
#                print(loss.shape)

#                print(x)
                #loss = -(x * value).mean()
                #loss = Variable(loss, requires_grad=True)
                total_loss += newLoss.item()
                # op.retain_grad()
                #print(loss)

                newLoss.backward()
                #print("backward1")
                self.optimizer.step()

                # when training size is large
#                if batchCnt % 2 == 0:
#                    newLoss.backward()
#                    #print("backward1")
#                    self.optimizer.step()
                num = num + 1

          logging.info("EPOCH Ends")
          print("Epoch{}".format(e))
          print("Epoch{} ::loss {} ->".format(e,total_loss/trainSize))
          grad_list[e] = total_loss
#          for param_group in self.optimizer.param_groups:
#            print(param_group['lr'])
#          if e > 1 and grad_list[e] == grad_list[e-1] and grad_list[e-1] == grad_list[e-2]:
#            break
          if e > 0 and grad_list[e] >= grad_list[e-1]:
            break
#          if total_loss > -200000:
#            break
#          else:
#            currentBestLoss = total_loss
#          if total_loss > -500:
#            break
#           print(self.val_loss(valid_econ, valid_prop))
          # print("______________")

        return e

    def val_loss(self, cap, feature, value):
        valueTemp = value.numpy()
#        test_instance = len(valueTemp) / self.batch_size
        test_instance = np.size(valueTemp, 0) / self.batch_size
#        itemVal = self.c.tolist()
        itemVal = self.c
        real_obj = actual_obj(itemVal, cap, value[:, 1], n_instance=int(test_instance))
#        print(np.sum(real_obj))

        self.model.eval()
        criterion = nn.L1Loss(reduction='mean')  # nn.MSELoss(reduction='sum')
        valid_df = MyCustomDataset(feature, value)
        valid_dl = data_utils.DataLoader(valid_df, batch_size=self.batch_size, shuffle=False)

        obj_list = []
        corr_obj_list = []

        feasible_sol_cost_list = list()
        feasible_sol_opt_cost_list = list()

        num_infeas_sol = 0

        len = np.size(valueTemp, 0)
        predVal = torch.zeros((len, 2))

        num = 0

        tot_squared_error = 0

        for feature, value in valid_dl:
            op = self.model(feature).squeeze()
#            print(op)
            loss = criterion(op, value)

            realWT = {}
            predWT = {}
            realPrice = {}
            predPrice = {}
            penaltyVector = np.zeros(itemNum)
            for i in range(itemNum):
                realWT[i] = value[i][1]
                predWT[i] = op[i][1]
                realPrice[i] = value[i][0]
                predPrice[i] = op[i][0]
                predVal[i+num*itemNum][0] = op[i][0]
                predVal[i+num*itemNum][1] = op[i][1]
                penaltyVector[i] = self.penalty[i+num*itemNum]

            corrrlst, feasible = correction_single_obj(realPrice, predPrice, cap, realWT, predWT, penaltyVector)

            if feasible:
                feasible_sol_cost_list.append(corrrlst)
                feasible_sol_opt_cost_list.append(real_obj[num])
            else:
                num_infeas_sol += 1

            real_price_as_tensor = torch.as_tensor(list(realPrice.values()))
            pred_price_as_tensor = torch.as_tensor(list(predPrice.values()))

            real_weights_as_tensor = torch.as_tensor(list(realWT.values()))
            pred_weights_as_tensor = torch.as_tensor(list(predWT.values()))

            preds = torch.cat((pred_price_as_tensor, pred_weights_as_tensor))
            real_vals = torch.cat((real_price_as_tensor, real_weights_as_tensor))
            squared_error = torch.square(preds - real_vals)
            squared_error = torch.sum(squared_error)
            tot_squared_error += squared_error

            corr_obj_list.append(corrrlst)
            num = num + 1

        self.model.train()
#        print(corr_obj_list-real_obj)
#        print(np.sum(corr_obj_list))
#        return prediction_loss, abs(np.array(obj_list) - real_obj)

        feasible_sol_cost_list = np.asarray(feasible_sol_cost_list)
        feasible_sol_opt_cost_list = np.asarray(feasible_sol_opt_cost_list)
        mse = torch.mean(tot_squared_error)
        mse = mse.detach().numpy()

        feas_sol_regret = abs(feasible_sol_cost_list - feasible_sol_opt_cost_list) / feasible_sol_opt_cost_list
        rel_regret = abs(np.array(corr_obj_list) - real_obj) / real_obj

        return rel_regret, predVal, feas_sol_regret, num_infeas_sol, mse

########################################################################################################################


if __name__ == '__main__':

    # Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--capacity', type=int, required=True)
    parser.add_argument('--penalty', type=int, required=True)
    parser.add_argument('--rnd-seed', type=int, required=True)

    # Parse the arguments
    args = parser.parse_args()

    capacity = args.capacity
    penaltyTerm = args.penalty
    rnd_seed = args.rnd_seed

    current_script_path = os.path.abspath(__file__)
    package_name = os.path.basename(os.path.dirname(current_script_path))
    loadpath_prefix = os.path.join('external', package_name, 'data', 'fractional_kp')

    print('Loading train prices...')
    loadpath = os.path.join(loadpath_prefix, 'train_prices', f'train_prices(' + str(testi) + ').txt')
    c_first_slice = np.loadtxt(loadpath)
    print('Finished')

    print('Loading train features...')
    loadpath = os.path.join(loadpath_prefix, os.path.join('train_features', f'train_features(' + str(testi) + ').txt'))
    x_first_slice = np.loadtxt(loadpath)
    print('Finished')

    y1_first_slice = c_first_slice.copy()

    print('Loading train weights...')
    loadpath = os.path.join(loadpath_prefix, 'train_weights', f'train_weights(' + str(testi) + ').txt')
    y2_first_slice = np.loadtxt(loadpath)
    print('Finished')

    print('Loading test prices...')
    loadpath = os.path.join(loadpath_prefix, 'test_prices', f'test_prices(' + str(testi) + ').txt')
    c_second_slice = np.loadtxt(loadpath)
    print('Finished')

    print('Loading test features...')
    loadpath = os.path.join(loadpath_prefix, 'test_features', f'test_features(' + str(testi) + ').txt')
    x_second_slice = np.loadtxt(loadpath)
    print('Finished')

    y1_second_slice = c_second_slice.copy()

    print('Loading test weights...')
    loadpath = os.path.join(loadpath_prefix, 'test_weights', f'test_weights(' + str(testi) + ').txt')
    y2_second_slice = np.loadtxt(loadpath)
    print('Elapsed')

    all_c = np.concatenate((c_first_slice, c_second_slice), axis=0)
    all_x = np.concatenate((x_first_slice, x_second_slice), axis=0)
    all_y1 = np.concatenate((y1_first_slice, y1_second_slice), axis=0)
    all_y2 = np.concatenate((y2_first_slice, y2_second_slice), axis=0)

    print('Loading train penalty...')
    loadpath = \
        os.path.join(loadpath_prefix,
                     'train_penalty' + str(penaltyTerm),
                     'train_penalty(' + str(testi) + ').txt')
    print('Finished')

    penalty_first_slice = np.loadtxt(loadpath)

    print('Loading test penalty...')
    loadpath = os.path.join(loadpath_prefix, 'test_penalty' + str(penaltyTerm), 'test_penalty(' + str(testi) + ').txt')
    penalty_second_slice = np.loadtxt(loadpath)
    print('Elapsed')

    all_penalty = np.concatenate((penalty_first_slice, penalty_second_slice), axis=0)

    #c_dataTemp = np.loadtxt('KS_c.txt')
    #c_data = c_dataTemp[:itemNum]

    h_data = np.ones(itemNum+1)
    h_data[itemNum] = capacity
    A_data = np.zeros((2, itemNum))
    b_data = np.zeros(2)


    #startmark = int(sys.argv[1])
    #endmark = startmark + 30

    print("*** HSD ****")

    #for testmark in range(startmark, endmark):
        #recordFile = open('record(' + str(testmark) + ').txt', 'a')
    recordBest = np.zeros((1, testTime))

    ReLUValue = 0
    if penaltyTerm == 0:
        ReLUValue = 15
    elif penaltyTerm == 0.25 or penaltyTerm == 0.5:
        ReLUValue = 23
    elif penaltyTerm == 1:
        ReLUValue = 24
    elif penaltyTerm == 2:
        ReLUValue = 26

    for run in range(testTime):
        testi = 0

        c_train, c_test, x_train, x_test, \
        y_train1, y_test1, y_train2, y_test2, \
        penalty_train, penalty_test = \
            train_test_split(all_c, all_x, all_y1, all_y2, all_penalty, random_state=rnd_seed)

        meanPriceValue = np.mean(y_train1)
        meanWeightValue = np.mean(y_train2)

        y_train = np.zeros((y_train1.size, 2))
        for i in range(y_train1.size):
            y_train[i][0] = y_train1[i]
            y_train[i][1] = y_train2[i]
        feature_train = torch.from_numpy(x_train).float()
        value_train = torch.from_numpy(y_train).float()

        y_test = np.zeros((y_test1.size, 2))
        for i in range(y_test1.size):
            y_test[i][0] = y_test1[i]
            y_test[i][1] = y_test2[i]
        feature_test = torch.from_numpy(x_test).float()
        value_test = torch.from_numpy(y_test).float()

        start = time.time()
        damping = 1e-2
        thr = 1e-3
        lr = 1e-7
        #    lr = 1e-5
        bestTrainCorrReg = float("inf")

        clf = Intopt(c_train, h_data, A_data, b_data, penalty_train, damping=damping, lr=lr, n_features=featureNum, thr=thr, epochs=8)
        train_epochs = clf.fit(feature_train, value_train)
        train_rslt, predTrainVal, train_feas_sol_regret, train_num_infeas_sol, train_mse = clf.val_loss(capacity, feature_train, value_train)
        avgTrainCorrReg = np.mean(train_rslt)
    #    trainHSD_rslt = str(testmark) + ' train: ' + str(np.sum(train_rslt[1])) + ' ' + str(np.mean(train_rslt[1]))
        trainHSD_rslt = 'train: ' + str(np.mean(train_rslt))

        if avgTrainCorrReg < bestTrainCorrReg:
            bestTrainCorrReg = avgTrainCorrReg
            torch.save(clf.model.state_dict(), 'model.pkl')
        print(trainHSD_rslt)

    #        if avgTrainCorrReg < 50:
    #            break

    #        val_rslt = clf.val_loss(source, sink, arc, feature_test, value_test)
    #        #HSD_rslt = str(testmark) + ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
    #        HSD_rslt = ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
    #        print(HSD_rslt)

    #    val_rslt = clf.val_loss(source, sink, arc, feature_test, value_test)
    ##    HSD_rslt = str(testmark) + ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
    #    HSD_rslt = ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
    #    print(HSD_rslt)
    #    print('\n')
    #    recordBest[0][i] = np.sum(val_rslt[1])

        clfBest = Intopt(c_test, h_data, A_data, b_data, penalty_test, damping=damping, lr=lr, n_features=featureNum, thr=thr, epochs=8)
        clfBest.model.load_state_dict(torch.load('model.pkl'))

        val_rslt, predTestVal, val_feas_sol_regret, val_num_infeas_sol, val_mse = clfBest.val_loss(capacity, feature_test, value_test)
        #HSD_rslt = str(testmark) + ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
        #    print(predTestVal.shape)
        end = time.time()

        predTestVal = predTestVal.detach().numpy()
        #    print(predTestVal.shape)
        predTestVal1 = predTestVal[:, 0]
        predTestVal2 = predTestVal[:, 1]
        predValuePrice = np.zeros((predTestVal1.size, 2))
        for i in range(predTestVal1.size):
        #        predValue[i][0] = int(i/itemNum)
            predValuePrice[i][0] = y_test1[i]
            predValuePrice[i][1] = predTestVal1[i]
        #    np.savetxt('./data/proposed_prices200/proposed_prices' + str(penaltyTerm) + '(' + str(testi) + ').txt', predValuePrice, fmt="%.2f")
        predValueWeight = np.zeros((predTestVal2.size, 2))
        for i in range(predTestVal2.size):
        #        predValue[i][0] = int(i/itemNum)
            predValueWeight[i][0] = y_test2[i]
            predValueWeight[i][1] = predTestVal2[i]
        #    np.savetxt('./data/proposed_weights200/proposed_weights' + str(penaltyTerm) + '(' + str(testi) + ').txt', predValueWeight, fmt="%.2f")

        HSD_rslt = 'test: ' + str(np.mean(val_rslt))
        savepath = os.path.join(f'capacity-{capacity}', f'penalty-{penaltyTerm}', f'rnd-split-{rnd_seed}')

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        n_test = len(val_rslt)
        val_rslt = np.mean(val_rslt)

        if len(val_feas_sol_regret) == 0:
            val_feas_sol_regret = -1
        else:
            val_feas_sol_regret = np.mean(val_feas_sol_regret)

        np.save(os.path.join(savepath, 'rel-regret.npy'), val_rslt)
        np.save(os.path.join(savepath, 'feas-sol-rel-regret.npy'), val_feas_sol_regret)
        np.save(os.path.join(savepath, 'num-infeas-sol.npy'), val_num_infeas_sol / n_test)
        np.save(os.path.join(savepath, 'train-epochs.npy'), train_epochs)
        np.save(os.path.join(savepath, 'mse.npy'), val_mse)

        print(HSD_rslt)
        print ('Elapsed time: ' + str(end-start))

