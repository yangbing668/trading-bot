import logging
import pandas as pd

# Formats Position
format_position = lambda price: ('-' if price < 0 else '+') + '{0:.2f}'.format(abs(price))

# Formats Currency
format_currency = lambda price: '{0:.2f}'.format(abs(price))


def show_train_result(result, val_position, maxAmount):
    """ Displays training results
    """
    if val_position == 0.0:
        logging.info(
            'Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f} Maximum expenditure: {:.4f}'
            .format(result[0], result[1], format_position(result[2]), result[3], result[4]))
    else:
        logging.info(
            'Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f} Maximum expenditure: {:.4f})'
            .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3],
                    maxAmount))


def show_eval_result(model_name, profit, maxAmount):
    """ Displays eval results
    """
    if profit == 0.0:
        logging.info('{}: USELESS\n'.format(model_name))
    else:
        logging.info(
            'Model Name:{}.h5: Total Profit:{} Maximum Expenditure:{}\n'.format(model_name, format_position(profit),
                                                                                maxAmount))


def get_stock_data(stock_file):
    """Reads stock data from csv file
    """
    df = pd.read_csv(stock_file)
    try:
        return list(df['close'])
    except:
        return list(df['Close'])


def get_stock(stock_file):
    try:
        return pd.read_csv(stock_file, index_col='date')
    except:
        return pd.read_csv(stock_file, index_col='Date')