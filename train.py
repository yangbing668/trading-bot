import logging
import coloredlogs
import os
from docopt import docopt

from trading_bot.agent import Agent
# from trading_bot.agent_MLP import Agent
from trading_bot.methods import train_model, evaluate_model
# from trading_bot.methods_MLP import train_model, evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_train_result
)
import random

random.seed(1234)


def main(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained=False,
         debug=False, memoryLenDefault=1000, serviceChargeRate=0.):
    agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name, memoryLenDefault=memoryLenDefault)
    train_data = get_stock_data(train_stock)
    val_data = get_stock_data(val_stock)
    # val_data_reversed = list(reversed(val_data))
    # val_data = val_data + val_data_reversed

    bestResult = -999999999999
    for episode in range(1, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count, batch_size=batch_size,
                                   window_size=window_size, purchasingAbility=purchasingAbility,
                                   serviceChargeRate=serviceChargeRate)
        val_result, history, maxAmount, total_profits = evaluate_model(agent, val_data, window_size, debug,

                                                                       purchasingAbility=purchasingAbility,
                                                                       serviceChargeRate=serviceChargeRate)
        if train_result[2] > bestResult:
            bestResult = train_result[2]
            agent.save()
        show_train_result(train_result, val_result, maxAmount)


if __name__ == "__main__":
    stock_names = os.listdir(dirs)
    random.shuffle(stock_names)
    for i in range(30):
        stock_name = stock_names[i].split('.')[0]
        print("++++++++++++++++++++++"+stock_name+"++++++++++++++++++++++")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定将要使用的GPU设备
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        serviceChargeRate = 0.001
        purchasingAbility = 500
        train_stock = dirs + stock_name + '.csv'
        val_stock = dirs + stock_name + '.csv'
        strategy = 'dqn'
        window_size = 12
        batch_size = 20
        ep_count = 50
        model_name = '{}_w{}_b{}_e{}'.format(stock_name, window_size, batch_size, ep_count)
        pretrained = False
        debug = False

        coloredlogs.install(level="DEBUG")
        # switch_k_backend_device()

        try:
            main(train_stock, val_stock, window_size, batch_size,
                 ep_count, strategy=strategy, model_name=model_name,
                 pretrained=pretrained, debug=debug, memoryLenDefault=10000,
                 serviceChargeRate=serviceChargeRate)
        except KeyboardInterrupt:
            print("Aborted!")
