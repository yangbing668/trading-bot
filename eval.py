import coloredlogs
from trading_bot.agent import Agent
from trading_bot.methods import evaluate_model, test_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_eval_result
)


def main(eval_stock, window_size, model_name, debug, purchasing_ability, serviceChargeRate):
    """ Evaluates the stock trading bot.
    """
    data = get_stock_data(eval_stock)
    agent = Agent(window_size, pretrained=True, model_name=model_name)
    profit, _, maxAmount, total_profits = evaluate_model(agent, data, window_size, debug, purchasing_ability, serviceChargeRate)
    # print(total_profits)
    show_eval_result(model_name, profit, maxAmount)

if __name__ == "__main__":
    import os
    dirs = 'models'
    stock_names = os.listdir(dirs)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 指定将要使用的GPU设备
    serviceChargeRate = 0.001
    purchasing_ability = 500
    for stock_name in stock_names:
        stock_name = stock_name.split('_')[0]
        eval_stock = 'downloadData/Data/test/'+stock_name+'.csv'
        strategy = 'dqn'
        window_size = 12
        batch_size = 20
        ep_count = 50
        model_name = '{}_w{}_b{}_e{}'.format(stock_name, window_size, batch_size, ep_count)
        debug = False
        coloredlogs.install(level="DEBUG")
        try:
            main(eval_stock, window_size, model_name, debug, purchasing_ability, serviceChargeRate)
        except KeyboardInterrupt:
            print("Aborted")
