import os
import logging

import numpy as np

from tqdm import tqdm

from .utils import (
    format_currency,
    format_position
)
from .ops import (
    get_state
)


def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10, purchasingAbility=5,
                serviceChargeRate=0.):
    total_profit = 0
    data_length = len(data) - 1
    agent.holdAmount = 0
    avg_loss = []

    state = get_state(data, 0, window_size + 1)
    holdShare = 0
    maxAmount = 0
    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):
        if agent.holdAmount > maxAmount:
            maxAmount = agent.holdAmount
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)
        done = (t == data_length - 1)
        # select an action
        action, actWeight = agent.act(state, is_eval=True)
        if done and holdShare > 0:
            bought_price = agent.holdAmount / holdShare
            serviceCharge = serviceChargeRate * agent.holdAmount
            # if data[t] > bought_price:
            delta = (data[t] - bought_price) * holdShare  # *(1-0.015)
            total_profit += delta
            total_profit -= serviceCharge
            reward = delta
            agent.holdAmount = 0
            purchasingAbility = purchasingAbility + (delta / data[0])
            # else:
            #     bought_price = agent.holdAmount / holdShare
            #     diff = data[t] - bought_price
            #     reward = diff * holdShare
            # pass
        # elif data[t] > np.mean(data[t-5:t-1]):
        #     action == 0
        else:
            # BUY
            if action == 1 and purchasingAbility - holdShare >= 100:
                bugShare = round((purchasingAbility - holdShare) * actWeight)
                bugShare = round(bugShare / 100) * 100
                if bugShare < 100:
                    bugShare = 100
                try:
                    bought_price = agent.holdAmount / holdShare
                    # diff = data[t] - bought_price
                    diff = bought_price - data[t]
                    reward = diff * bugShare
                except:
                    pass
                holdShare += bugShare
                agent.holdAmount += data[t] * bugShare
                serviceCharge = data[t] * bugShare * serviceChargeRate
                total_profit -= serviceCharge

                # reward = -serviceCharge
            # SELL
            elif action == 2 and holdShare > 0:
                bought_price = agent.holdAmount / holdShare
                diff = data[t] - bought_price
                # if diff > 0:
                sellShare = round(holdShare * actWeight)
                sellShare = round(sellShare / 100) * 100
                if sellShare < 100 and holdShare >= 100:  # 最小卖出份额为1
                    sellShare = 100
                if holdShare < 100:
                    sellShare = holdShare
                holdShare -= sellShare
                agent.holdAmount -= sellShare * bought_price
                delta = diff * sellShare
                serviceCharge = sellShare * data[t] * serviceChargeRate
                total_profit -= serviceCharge
                reward = delta  # max(delta, 0)
                total_profit += delta
                purchasingAbility = purchasingAbility + (delta / data[0])
            # HOLD
            elif holdShare > 0:
                bought_price = agent.holdAmount / holdShare
                diff = data[t] - bought_price
                reward = diff * holdShare
            else:
                pass
        agent.remember(state, action, reward, next_state, done, actWeight)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    # if episode % 10 == 0:
    #     agent.save()
    # agent.reset()
    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)), purchasingAbility * data[0])


def evaluate_model(agent, data, window_size, debug, purchasingAbility=5, serviceChargeRate=0.):
    total_profit = 0
    data_length = len(data) - 1
    total_profits = []
    history = []
    agent.holdAmount = 0
    holdShare = 0
    state = get_state(data, 0, window_size + 1)
    maxAmount = purchasingAbility * data[0]
    for t in range(data_length):
        # if agent.holdAmount > maxAmount:
        #     maxAmount = agent.holdAmount
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)
        done = (t == data_length - 1)
        # select an action
        action, actWeight = agent.act(state, is_eval=True)
        # action, actWeight = 1, 1
        if done:
            if holdShare > 0:
                serviceCharge = serviceChargeRate * agent.holdAmount
                bought_price = agent.holdAmount / holdShare
                # if data[t] > bought_price:
                delta = (data[t] - bought_price) * holdShare  # *(1-0.015)
                purchasingAbility = purchasingAbility + (delta / data[0])
                sellShare = holdShare
                holdShare -= sellShare
                total_profit += delta
                reward = delta
                total_profit -= serviceCharge
                history.append((data[t], "SELL", sellShare))
                if debug:
                    logging.debug("Sell at: {} | Position: {}; sellShare: {}; holdShare: {}; profit: {}".format(
                        format_currency(data[t]), format_position(data[t] - bought_price), sellShare, holdShare,
                        format_position(delta)))
                agent.holdAmount = 0
                # else:
                #     bought_price = agent.holdAmount / holdShare
                #     diff = data[t] - bought_price
                #     reward = diff * holdShare
                #     history.append((data[t], "HOLD", 0))
                #     if debug:
                #         logging.debug("Hold at: {} | Position: {}; holdShare: {} holdAmount: {}".format(
                #             format_currency(data[t]), format_position(data[t] - bought_price), holdShare, agent.holdAmount))
            else:
                history.append((data[t], "HOLD", 0))
        # elif data[t] > np.mean(data[t - 5:t - 1]):
        #     history.append((data[t], "HOLD", 0))
        #     action == 0
        else:
            # BUY
            if action == 1 and purchasingAbility - holdShare >= 100:
                bugShare = round((purchasingAbility - holdShare) * actWeight)
                bugShare = round(bugShare / 100) * 100
                if bugShare < 100:
                    bugShare = 100
                holdShare += bugShare
                agent.holdAmount += data[t] * bugShare
                history.append((data[t], "BUY", bugShare))
                serviceCharge = data[t] * bugShare * serviceChargeRate
                try:
                    bought_price = agent.holdAmount / holdShare
                    diff = bought_price - data[t]
                    reward = diff * bugShare
                except:
                    pass
                total_profit -= serviceCharge
                if debug:
                    logging.debug(
                        "Buy at: {}; bugShare: {}; holdShare: {}; profit: {}".format(format_currency(data[t]), bugShare,
                                                                                     holdShare, format_position(
                                0 - serviceCharge)))

            # SELL
            elif action == 2 and holdShare > 0:
                bought_price = agent.holdAmount / holdShare
                diff = data[t] - bought_price
                # if diff > 0:
                sellShare = round(holdShare * actWeight)
                sellShare = round(sellShare / 100) * 100
                if sellShare < 100 and holdShare >= 100:  # 最小卖出份额为1
                    sellShare = 100
                if holdShare < 100:
                    sellShare = holdShare
                holdShare -= sellShare
                agent.holdAmount -= sellShare * bought_price
                delta = diff * sellShare
                purchasingAbility = purchasingAbility + (delta / data[0])
                reward = delta  # max(delta, 0)
                total_profit += delta
                serviceCharge = sellShare * data[t] * serviceChargeRate
                total_profit -= serviceCharge
                history.append((data[t], "SELL", sellShare))
                if debug:
                    logging.debug("Sell at: {} | Position: {}; sellShare: {}; holdShare: {}; profit: {}".format(
                        format_currency(data[t]), format_position(data[t] - bought_price), sellShare, holdShare,
                        format_position(delta - serviceCharge)))
                # else:
                #     bought_price = agent.holdAmount / holdShare
                #     diff = data[t] - bought_price
                #     reward = diff * holdShare
                #     history.append((data[t], "HOLD", 0))
            # HOLD
            else:
                # if holdShare > 0:
                #     bought_price = agent.holdAmount / holdShare
                #     diff = data[t] - bought_price
                #     reward = diff * holdShare
                # pass
                history.append((data[t], "HOLD", 0))
                # if holdShare > 0:
                #     bought_price = agent.holdAmount / holdShare
                #     diff = data[t] - bought_price
                #     total_profit += (diff * holdShare)
        # agent.remember(state, action, reward, next_state, done)
        # total_profits.append(total_profit)
        if total_profit == 0 and agent.holdAmount > 0:
            bought_price = agent.holdAmount / holdShare
            total_profits.append(agent.holdAmount * (data[t] - bought_price) / bought_price)
        else:
            if holdShare == 0:
                total_profits.append(total_profit)
            else:
                bought_price = agent.holdAmount / holdShare
                total_profits.append(total_profit * (1 + (data[t] - bought_price) / bought_price))
        # agent.memory.append((state, action, reward, next_state, done))
        state = next_state
    return total_profit, history, maxAmount, total_profits


def test_model(agent, data, window_size, debug, purchasingAbility=5, serviceChargeRate=0.):
    total_profit = 0
    data_length = len(data) - 1
    total_profits = []
    history = []
    agent.holdAmount = 0
    holdShare = 0
    state = get_state(data, 0, window_size + 1)
    maxAmount = 0
    for t in range(data_length):
        if agent.holdAmount > maxAmount:
            maxAmount = agent.holdAmount
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)
        done = (t == data_length - 1)
        # select an action
        action, actWeight = agent.act(state, is_eval=True)
        if done:
            if agent.holdAmount > 0:
                serviceCharge = serviceChargeRate * agent.holdAmount
                bought_price = agent.holdAmount / holdShare
                if data[t] > bought_price:
                    delta = (data[t] - bought_price) * holdShare  # *(1-0.015)
                    sellShare = holdShare
                    holdShare -= sellShare
                    total_profit += delta
                    reward = delta
                    total_profit -= serviceCharge
                    history.append((data[t], "SELL", sellShare))
                    if debug:
                        logging.debug("Sell at: {} | Position: {}; sellShare: {}; holdShare: {}".format(
                            format_currency(data[t]), format_position(data[t] - bought_price), sellShare, holdShare))
                    agent.holdAmount = 0
                else:
                    bought_price = agent.holdAmount / holdShare
                    diff = data[t] - bought_price
                    reward = diff * holdShare
                    history.append((data[t], "HOLD", 0))
                    if debug:
                        logging.debug("Hold at: {} | Position: {}; holdShare: {} holdAmount: {}".format(
                            format_currency(data[t]), format_position(data[t] - bought_price), holdShare,
                            agent.holdAmount))
            else:
                # pass
                history.append((data[t], "HOLD", 0))
        # elif data[t] > np.mean(data[t - 5:t - 1]):
        #     history.append((data[t], "HOLD", 0))
        #     action == 0
        else:
            # BUY
            if action == 1 and purchasingAbility - holdShare >= 100:
                bugShare = round((purchasingAbility - holdShare) * actWeight)
                if bugShare < 100:
                    bugShare = 100
                holdShare += bugShare
                agent.holdAmount += data[t] * bugShare
                history.append((data[t], "BUY", bugShare))
                serviceCharge = data[t] * bugShare * serviceChargeRate
                reward = -serviceCharge
                total_profit -= serviceCharge
                if debug:
                    logging.debug(
                        "Buy at: {}; bugShare: {}; holdShare: {}".format(format_currency(data[t]), bugShare, holdShare))

            # SELL
            elif action == 2 and holdShare > 0:
                bought_price = agent.holdAmount / holdShare
                diff = data[t] - bought_price
                if diff > 0:
                    sellShare = round(holdShare * actWeight)
                    if sellShare < 100 and holdShare >= 100:  # 最小卖出份额为1
                        sellShare = 100
                    holdShare -= sellShare
                    agent.holdAmount -= sellShare * bought_price
                    delta = diff * sellShare
                    reward = delta  # max(delta, 0)
                    total_profit += delta
                    serviceCharge = sellShare * data[t] * serviceChargeRate
                    total_profit -= serviceCharge
                    history.append((data[t], "SELL", sellShare))
                    if debug:
                        logging.debug("Sell at: {} | Position: {}; sellShare: {}; holdShare: {}".format(
                            format_currency(data[t]), format_position(data[t] - bought_price), sellShare, holdShare))
                else:
                    if holdShare > 0:
                        bought_price = agent.holdAmount / holdShare
                        diff = data[t] - bought_price
                        reward = diff * holdShare
                    # pass
                    history.append((data[t], "HOLD", 0))
            # HOLD
            else:
                if holdShare > 0:
                    bought_price = agent.holdAmount / holdShare
                    diff = data[t] - bought_price
                    reward = diff * holdShare
                # pass
                history.append((data[t], "HOLD", 0))
        agent.remember(state, action, reward, next_state, done, actWeight)
        if total_profit == 0 and agent.holdAmount > 0:
            bought_price = agent.holdAmount / holdShare
            total_profits.append(agent.holdAmount * (data[t] - bought_price) / bought_price)
        else:
            if holdShare == 0:
                total_profits.append(total_profit)
            else:
                bought_price = agent.holdAmount / holdShare
                total_profits.append(total_profit * (1 + (data[t] - bought_price) / bought_price))
        # agent.memory.append((state, action, reward, next_state, done))
        state = next_state
    return total_profit, history, maxAmount, total_profits
