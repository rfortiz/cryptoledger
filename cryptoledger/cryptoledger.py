#!/usr/bin/env python3

"""
Cryptoledger is a tool to keep track of cryptocurrency transactions

* Imports *.csv transaction lists exported from exchanges (bittrex and gdax supported) and consolidates them in a standardize *.csv file without duplicates.
* Calculate a portfolio on any given dates using the transaction list
* Estimate its value in specified currency using cryptocompare API
* Plot a portfolio pie chart, daily valuation and daily return for a given period
* Create a corresponding PDF report formatted with a Latex template
"""

import sys # system specific package (e.g. provide argv for command line arguments)
import argparse # command line parser
import csv # read and write csv file
import matplotlib.pyplot as plt # for plotting graph
import seaborn as sns # for plotting graphs, nicer theme for matplotlib + wrapper for stat related plot
import datetime as dt # to convert strings from csv to date and time
import calendar # to conver date time to unix timestamp
import warnings
import os # to check if a file exist
import numpy as np # handle multi-dimensional arrays
import pandas as pd # data manipulation and analysis
from pandas.util import hash_pandas_object
import requests # to get json files from cryptocompare API
import jinja2 # to use a latex template
from jinja2 import Template
import shutil # file operations (e.g. copy)

class Transaction:
    """Class representing a transaction (BUY|SELL|WITHDRAWAL|DEPOSIT), converts exchanges raw csv to internal 'native' format"""
    
    fieldnames = ['uid', 'base_c', 'quote_c', 'action', 'qty', 'rate', 'amount', 'commission', 'timestamp', 'exchange'] # short fieldnames for internal use
    fieldnames_exp = ['UID', 'Base currency', 'Quote currency', 'Action', 'Quantity (#base)', 'Rate (Quote currency)', 'Amount (Quote currency)', 'Commission (Quote currency)', 'Timestamp', 'Exchange'] # explicit fieldnames for export

    def __init__(self, uid='', base_c='', quote_c='', action='', qty=0.0, rate=0.0, amount=0.0, commission=0.0, timestamp=0, exchange=''):
        """initialize transaction with default value"""
        
        self.uid = uid # uniqueID (keep original coin transfer ID or exchange trade id for Traceability)
        self.base_c = base_c # base currency (e.g. alts)
        self.quote_c = quote_c # quote currency (e.g. BTC, EUR)
        self.action = action # action (BUY|SELL|WITHDRAW|DEPOSIT)
        self.qty = qty # qty (of base)
        self.rate = rate # rate (quote currency/base currency)
        self.amount = amount # amount (in quote currency, excluding commission)
        self.commission= commission # commission (in quote currency)
        self.timestamp = timestamp # timestamp (of exectued time)
        self.exchange = exchange # name of exchange
        
    def __str__(self):
        """returns text representation of a transaction"""
        return self.fieldnames_exp[0]+" = "+self.uid+"\n"+ \
                self.fieldnames_exp[1]+" = "+self.base_c+"\n"+ \
                self.fieldnames_exp[2]+" = "+self.quote_c+"\n"+ \
                self.fieldnames_exp[3]+" = "+self.action+"\n"+ \
                self.fieldnames_exp[4]+" = "+str(self.qty)+"\n"+ \
                self.fieldnames_exp[5]+" = "+str(self.rate)+"\n"+ \
                self.fieldnames_exp[6]+" = "+str(self.amount)+"\n"+ \
                self.fieldnames_exp[7]+" = "+str(self.commission)+"\n"+ \
                self.fieldnames_exp[8]+" = "+ dt.datetime.utcfromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S') +"\n"+ \
                self.fieldnames_exp[9]+" = "+self.exchange+"\n"
    
    def to_dict(self):
        """Returns a transaction in a dict of strings with formated numbers"""
        t = dt.datetime.utcfromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        row = {}
        row['uid'] = self.uid
        row['base_c'] = self.base_c
        row['quote_c'] = self.quote_c
        row['action'] = self.action
        row['qty'] = '{:.8g}'.format(self.qty)
        row['rate'] = '{:.8g}'.format(self.rate)
        row['amount'] = '{:.8g}'.format(self.amount)
        row['commission'] = '{:.8g}'.format(self.commission)
        row['timestamp'] = t
        row['exchange'] = self.exchange
        return row
    
    def __eq__(self, other):
        """delegate __eq__ to tuple"""
        return self.to_tuple() == other.to_tuple()

    def __hash__(self):
        """delegate __hash__ to tuple"""
        return hash(self.to_tuple())
        
    def to_tuple(self):
        """return attributes defining a transaction as a list"""
        return (self.uid, self.base_c, self.quote_c, self.action, self.qty, self.rate, self.amount, self.commission, self.timestamp, self.exchange)
    
    def is_valid(self): # TODO: test other conditions and issue error/warning printing availalbe info
        """test if a transaction is valid (mendatory fields depending on action, verifiy qty*rate == amount, etc.)"""
        return self.qty*self.rate == self.amount
    
    @classmethod
    def from_dict(cls, dic, d_format='native'):
        """create a transaction from a line of CSV file read by DictReader."""
        
        if d_format == 'native': # i.e. from csv database saved by this program
            #~ return cls(*dic.values()) # '*' expand the content of the list as arguments, assumes the order is the same as defined in init function; need to normalize types for sort and hash operations
            return cls(dic['uid'], dic['base_c'], dic['quote_c'], dic['action'], float(dic['qty']), float(dic['rate']), float(dic['amount']), float(dic['commission']), int(dic['timestamp']), dic['exchange'])
            
        elif d_format == 'bittrex':
            # extract base and quote currencies
            (q_c,b_c) = dic['Exchange'].split('-') # assumes there is only one '-'
            
            # check for valid action
            act = str()
            if 'SELL' in dic['Type'].upper():
                act = 'SELL'
            elif 'BUY' in dic['Type'].upper():
                act = 'BUY'
            else:
                act = 'unknown action' + dic['Type'] # not that withdrawal and deposit do not appear in bittrex csv export, need to be added manually
            
            #convert datetime string to timestamp
            date_conv = dt.datetime.strptime(dic['Closed'],'%m/%d/%Y %I:%M:%S %p') # bittrex datetime format, e.g. 12/5/2017 1:46:32 PM, assumes UTC time zone
            date_timestamp = calendar.timegm(date_conv.utctimetuple()) # assumes UCT time, alternatively mktime uses localtiem instead
            
            return cls(uid=dic['OrderUuid'], base_c=b_c, quote_c=q_c, action=act, qty=float(dic['Quantity']), rate=float(dic['Limit']), amount=float(dic['Price']), commission=float(dic['CommissionPaid']), timestamp=date_timestamp, exchange='bittrex')
        
        elif d_format == 'gdax':
            # extract base and quote currencies
            (b_c,q_c) = dic['product'].split('-') # assumes there is only one '-'
            
            #convert datetime string to timestamp
            date_conv = dt.datetime.strptime(dic['created at'][:-5],'%Y-%m-%dT%H:%M:%S') # gdax datetime format, e.g. 2017-12-05T12:41:32.165Z, assumes always Z (UTC time zone), ignore microseconds
            date_timestamp = calendar.timegm(date_conv.utctimetuple()) # assumes UCT time, alternatively mktime uses localtiem instead
            
            #calculate amount excluding fees (standardized data)
            qty = float(dic['size'])
            rate = float(dic['price'])
            amount=qty*rate
            
            return cls(uid=dic['trade id'], base_c=b_c, quote_c=q_c, action=dic['side'], qty=qty, rate=rate, amount=amount , commission=float(dic['fee']), timestamp=date_timestamp, exchange='gdax')
            
        else:
            warnings.warn('Unable to init transaction from dict; Unsupported exchange: ' + str(source), UserWarning)
            return(cls)
            
    @classmethod
    def from_input(cls):
        """create a transaction from manual user input, return None if canceled or invalid"""
        #could be improve by validating input one by one or better way to input timestamp, prefill depending on action, etc.
        user_input = []
        tr = None
        while True:
            user_input.clear()
            print('Manual transaction input', 'timestamp format: %Y-%m-%d %H:%M:%S','', sep='\n')
            try:
                for line in cls.fieldnames_exp:
                    user_input.append(input(line+': '))
                date_conv = dt.datetime.strptime(user_input[8],'%Y-%m-%d %H:%M:%S') 
                date_timestamp = calendar.timegm(date_conv.utctimetuple())
                tr = cls(user_input[0], user_input[1], user_input[2], user_input[3], float(user_input[4]), float(user_input[5]), float(user_input[6]), float(user_input[7]), date_timestamp, user_input[9])
                print('\nNew transaction:',tr,sep='\n')
            except ValueError:
                print('Invalid input.')
                if input('type "y" to try again: ') != 'y':
                    return None
            else:
                if input("Enter to confirm, type 'c' to cancel: ") == 'c':
                   return None
                else:
                     return tr
        return None
            
class Ledger:
    """Class representing a ledger containing a list of transactions. Add transactions from external csv, remove duplicates and sort"""
    
    def __init__(self, csv_db_path ):
        """Initialize the ledger with transactions in csv given in argument if it exist"""
        self.transactions = [] # list of transactions
        self.csv_db_path = csv_db_path
        if os.path.isfile(self.csv_db_path): # load db if it exists
            self.append(self.csv_db_path, 'native')

        
    def __str__(self):
        """returns text representation of a ledger"""
        text = 'list of transactions:'
        for line in self.transactions:
            text += '\n' + str(line)
            
        return text
        
    def append(self, csv_path, csv_format):
        """read transactions from a CSV file, add them to current list and remove duplicates"""
        with open(csv_path,'r',newline='\n', encoding='utf-8') as csv_update:
            csv_update = (line.replace('\0','') for line in csv_update)# use inline generator to replace NULL values (assumes NULL values are not due to incorrect encoding)
            csv_reader = csv.DictReader(csv_update, Transaction.fieldnames if csv_format=='native' else None)
            
            if csv_format == 'native':
                next(csv_reader) # skip the first line when using native format because headers are given separately
                
            for line in csv_reader:
                self.transactions.append( Transaction.from_dict(line, csv_format) )
            self.remove_duplicates()
            
    def manual_append(self):
        """Manually append new transaction"""
        tr = Transaction.from_input()
        if tr is not None:
            self.transactions.append(tr)
            self.remove_duplicates()
        
    def save(self):
        with open(self.csv_db_path,'w', encoding='utf-8') as csv_ledger:
            csv_writer = csv.writer(csv_ledger,delimiter=',')
            csv_writer.writerow(Transaction.fieldnames_exp) # write header line with expanded names
            for line in self.transactions:
                csv_writer.writerow( line.to_tuple() )
    
    def to_list(self, first_date, last_date):
        """Returns a list of transactions between the first and last date, formatted as a list of dict"""
        
        table = []
        for l in self.transactions:
            if first_date <= dt.datetime.utcfromtimestamp(l.timestamp).date() <= last_date:
                table.append(l.to_dict())
        return table
            
    def remove_duplicates(self):
        """Remove duplicates from transaction list and resort according to timestamp"""
        self.transactions = list(set(self.transactions))
        self.transactions = sorted(self.transactions, key=lambda k: k.timestamp,reverse=False) # sort list of dict according to key
        
class Portfolio:
    """Class calculating a portfolio as function of the date from a ledger (list of transactions)"""
    
    def __init__(self, ledger, eval_symbol='EUR'):
        """ initialize portfolio with ledger pass in argument."""
        self.ledger = ledger
        self.snapshot = None # portfolio snapshot on last date
        self.p_raw = pd.DataFrame() # portfolio with quantity of each coin
        self.calculate(ledger)
        
        self.eval_symbol=eval_symbol # currency in which to evalute the portfolio
        self.p_eval = pd.DataFrame() # portfolio evaluated in eval_symbol currency 
        self.evaluate()
        
    def calculate(self, ledger):
        """Calculate portfolio from ledger"""
        if len(ledger.transactions) == 0:
            return
        
        #populate portfolio by processing the transactions in ledger
        for l in ledger.transactions:
            day = pd.to_datetime(l.timestamp,unit='s', utc=True, origin='unix' ).date() # convert timestamp to datetime and keep the date part
            
            # if new day, copy last line (if exist) in dataframe with new day as index
            if day not in self.p_raw.index and len(self.p_raw) > 0:  
                self.p_raw = self.p_raw.append( pd.DataFrame(data=self.p_raw.tail(1).values, index=[day], columns=self.p_raw.columns) )
    
            # if base currency is not in portfolio yet, initilize to zero
            if l.base_c not in self.p_raw.columns: 
                self.p_raw.at[day,l.base_c] = 0.0
            # if base currency at current index doesn't exist, initialize to zero (should not happen if new liens are copies of previous)
            elif pd.isna(self.p_raw.at[day,l.base_c]): 
                self.p_raw.at[day,l.base_c] = 0.0
            # same for quote currency
            if l.quote_c not in self.p_raw.columns: 
                self.p_raw.at[day,l.quote_c] = 0.0
            elif pd.isna(self.p_raw.at[day,l.base_c]): 
                self.p_raw.at[day,l.quote_c] = 0.0
            
            # do actual calculations for current transaction
            if l.action == 'BUY':
                self.p_raw.at[day,l.base_c] += l.qty
                self.p_raw.at[day,l.quote_c] -= l.amount
                self.p_raw.at[day,l.quote_c] -= l.commission
            elif l.action == 'SELL':
                self.p_raw.at[day,l.base_c] -= l.qty 
                self.p_raw.at[day,l.quote_c] += l.amount
                self.p_raw.at[day,l.quote_c] -= l.commission
            elif l.action =='WITHDRAW':
                self.p_raw.at[day,l.quote_c] -= l.qty
                self.p_raw.at[day,l.quote_c] -= l.commission
            elif l.action =='DEPOSIT':
                self.p_raw.at[day,l.quote_c] += l.qty
                self.p_raw.at[day,l.quote_c] -= l.commission
        
        #make the index go until today, add missing rows (days when no transaction happend) and rename index
        self.p_raw = self.p_raw.reindex(pd.date_range(start=self.p_raw.index[0], end=pd.to_datetime('today').date(), freq='D'))
        self.p_raw.index.name = 'Day'
        
        # fill NaN values with previous valid ones or zeros if no previous values
        self.p_raw.fillna(method='ffill', inplace=True) # forward fill propagate[s] last valid observation forward to next valid
        self.p_raw.fillna(0.0, inplace=True) # replace remaining NaN by zeros
    
        # roundoff values close to zero (12 decimals) but keep full precision for the rest  (warning! float comparison only safe for zeros)
        self.p_raw=self.p_raw.where( self.p_raw.round(12) != 0.0, 0.0) # 'where' replace anything that does NOT satisfy condition
        
        
    def evaluate(self):
        """ Use cryptocomapre API to evaluate portfolio in given currency """
        # load backup file if exist and compare hash of previous p_raw to current p_raw before getting historical data cryptocompare API (slow)
        
        # create a tmp dir in the same folder as the as the ledger csv
        tmp_dir = os.path.dirname(os.path.abspath(self.ledger.csv_db_path)) + '/tmp' 
        if not os.path.exists(tmp_dir):  # create the tmp directory if not existing
            os.makedirs(tmp_dir)
        
        file_path = tmp_dir + '/p_eval_' + self.eval_symbol + '.pkl'
        hash_path = tmp_dir + '/p_raw_hash_' + self.eval_symbol
        current_hash = str(hash_pandas_object(self.p_raw).sum())
        old_hash = ''
        is_loaded = True
        # NOTE: better way would be to check if path is valid (although doesn't guarantee it can be open)
        try:
            self.p_eval = pd.read_pickle(file_path) #try to load previous file back
            with open(hash_path) as f:  
                old_hash = f.read() 
        except IOError:
            # If fail,  evalulate the portfolio and save the file after
            is_loaded = False
            
        if not is_loaded or old_hash != current_hash: # file was open but is not up to date (note that new transaction will only be updated on next day)
            self.p_eval = self.p_raw.apply(Portfolio.valuation, args=(self.eval_symbol,))
            self.p_eval.to_pickle(file_path)
            with open(hash_path, "w") as f:
                f.write(current_hash)
        
        
        return self.p_eval
        
    def snapshot_to_latex(self):
        """Returns a latex string of the portfolio snapshot after adding a lineseparator before the last line to the pandas.to_latex() output"""
        if self.snapshot is not None:
            latex = self.snapshot.to_latex(float_format='%.5g')
            latex_list = latex.splitlines()
            latex_list.insert(len(latex_list)-3, '\midrule')
            latex_new = '\n'.join(latex_list)
            return latex_new
        return ''
        
        
    def plot_return(self, start='first', end='last'):
        """Plots portfolio valuation vs day and daily return"""
        sns.set_style('ticks') # possible options white, dark, whitegrid, darkgrid, and ticks
        
        # subset of portfolio according to date range
        first_date = None
        last_date = None
        
        first_date = self.str_date_check(start)
        if first_date is None:
            return None 
            
        last_date = self.str_date_check(end)
        if last_date is None:
            return None
        
        p_sub = self.p_eval[first_date:last_date]
        
        # plot portfolio valuation and daily return with double y axis using matplotlib
        tot = p_sub.sum(axis=1) # sum all rows for total valuation on each day; dataSerie, contains index and sumation
        daily_return = 100*tot.pct_change(1) # calculate daily percent change
    
        fig = plt.figure(figsize=(10,6), dpi=100)
        ax1 = fig.add_subplot(111)
        ax1.plot(tot, label='Valuation')
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel('Time (day)')
        ax1.set_ylabel('Valuation (' + self.eval_symbol + ')')
        ax1.set_title(' Porfolio valuation in '+ self.eval_symbol, fontsize='x-large', fontweight='bold')
        ax2 = ax1.twinx()
        ax2.bar(daily_return.index, daily_return.values, label='Daily return', color='black', alpha=0.2)
        ax2.set_ylabel('Daily return (%)')
        fig.legend(bbox_transform=ax1.transAxes, bbox_to_anchor=[0.0, 1], loc='upper left') # use figure legend (combines legends of both y axis) and locate with respect to axis (instead of figure)
        plt.figtext(0.62, 0.22,'Average daily return: '+str( daily_return.mean().round(2))+'%')
        fig.autofmt_xdate() # auto rotate dates to try prevent overlapping
        #~ fig.tight_layout()
        
        return fig
        
    def plot_valuation(self, date='last'):
        """Pie plots of portfolio valuation on specified date (aslo print portfolio value in eval__symbol)"""
        sns.set_style('white') # possible options white, dark, whitegrid, darkgrid, and ticks
        
        # set the date
        d = self.str_date_check(date)
        if d is None:
            return None 
        
        # get portfolio snapshot on specified day
        snapshot=self.p_eval.loc[d]
        snapshot=snapshot[snapshot > 0.0] # drops zeros and negative values (can happen with incomplete ledger) so that labels dont's show
        snapshot.name = self.eval_symbol # only keep the date from name (datetime)
        
        # plot portfolio allocation using pandas wrapper
        fig = plt.figure(figsize=(4,4), dpi=120)
        ax = (snapshot*1000).plot.pie(autopct='%.2f%%', pctdistance=0.8) # *1000: hack to force drawing full circle when sum < 1
        #~ ax.set_title(' Portfolio allocation', fontsize='x-large', fontweight='bold')
        ax.set_ylabel('')
        #~ ax.set_xlabel(snapshot.name, fontweight='bold') # the name of the serie corresponds to index of the portfolio dataframe i.e. datetime type
        plt.axis('equal')
        #~ fig.tight_layout()
        
        #calculate portfolio sum and update in self
        snapshot['Total'] = snapshot.sum() # calculate and add total
        self.snapshot = snapshot

        return fig
        
    def str_date_check(self, s):
        """Check if a string is a valid date, if so also check that it is available in index"""
        d = None
        if s == 'last':
            d = self.p_eval.index[-1].date()
            return d
        elif s == 'first':
            d = self.p_eval.index[0].date()
            return d
        
        # check if the string s is a valid date
        try:
                d = pd.to_datetime(s).date()
        except ValueError:
            warnings.warn('Invalid date. expected format: YYYY-MM-DD, received: ' + s, UserWarning)
            return None
            
        # check if the converted date is in the index
        if d not in self.p_eval.index:
                warnings.warn('Requested date: ' + s + ' is not available.', UserWarning)
                return None
        return d
        
    @staticmethod
    def valuation(s, quote='EUR'):
        """function to to use with DataFrame.apply; evaluate the received dataserie in specified currency"""
        base = s.name
        if base == quote:
            return s
        print('calling cryptocompare API to convert', base, 'to', quote)
        time_s = calendar.timegm(s.index[-1].utctimetuple()) # convert last index value to timestamp
        duration = len(s)-1 # number of days to pull from API
        request_str = 'https://min-api.cryptocompare.com/data/histoday?fsym=' + base + '&tsym=' + quote + '&limit=' + str(duration) + '&toTs=' + str(time_s)
        df = pd.DataFrame(requests.get(request_str).json()['Data']) # pull data from cryptocompare api
        return s*df['close'].values
        
class Report:
    """Class generating a pdf report from a Portfolio using a Latex template. It includes portfolio valuation, daily return and transaction list."""
    
    def __init__(self, portfolio, report_folder, start='first', end='last'):
        self.portfolio = portfolio
        self.start = start
        self.end = end
        
        # create a report folder in the same folder as the as the ledger csv
        self.report_folder = os.path.dirname(os.path.abspath(self.portfolio.ledger.csv_db_path)) + '/report'
        if not os.path.exists(self.report_folder):  # create the tmp directory if not existing
            os.makedirs(self.report_folder)
        
        # create a build subfolder if it doesn't exist
        self.build_folder = self.report_folder + '/build'
        if not os.path.exists(self.build_folder):  # create the build directory if not existing
            os.makedirs(self.build_folder)
            
        # create the latex template with jinja2, template is expected in same folder as this python script
        template_Path = os.path.dirname(os.path.realpath(__file__)) + '/report_template.tex'
        self.template = Report.get_template(template_Path)
        
    def generate(self):
        """ Generate report variables (plot and tables) and render the template"""
        # create a dictionary with latex snippet for tables and path to plot saved as pdf 
        # try to use \BLOCK to create transactions table
        
        # generate and save plots as pdf
        portfolio_fig = self.portfolio.plot_valuation(date=self.end)
        portfolio_fig_filename = self.build_folder + '/portfolio.pdf'
        portfolio_fig.savefig(portfolio_fig_filename, format='pdf')
        
        return_fig = self.portfolio.plot_return(start=self.start, end=self.end)
        return_fig_filename = self.build_folder + '/return.pdf'
        return_fig.savefig(return_fig_filename, format='pdf')
        
        # create a dict of variables used in report template
        first_date = self.portfolio.str_date_check(self.start)
        last_date =self.portfolio.str_date_check(self.end)
        
        report_vars = {}
        report_vars['portfolio_fig'] = portfolio_fig_filename
        report_vars['return_fig'] = return_fig_filename
        report_vars['eval_symbol'] = self.portfolio.eval_symbol
        report_vars['start_date'] = first_date
        report_vars['end_date'] = last_date
        report_vars['portfolio_snapshot'] = self.portfolio.snapshot_to_latex() # uses pandas to_latex function with an extra \midrule before last line. For more control over latex formatting could use Jinja2 with for loop block as done for the transactions list in ledger
        report_vars['ledger_header'] = Transaction.fieldnames_exp
        report_vars['ledger_table'] = self.portfolio.ledger.to_list(first_date, last_date)
        
        with open(self.build_folder + '/report.tex', "w") as f:
            f.write(self.template.render(**report_vars))
        
    def compile_pdf(self):
        """Compile the Latex file with pdflatex"""
        
        os.system('pdflatex -synctex=1 -interaction=nonstopmode -output-directory {} {}'.format(self.build_folder, self.build_folder + '/report.tex'))
        os.system('pdflatex -synctex=1 -interaction=nonstopmode -output-directory {} {}'.format(self.build_folder, self.build_folder + '/report.tex')) # compile latex twice for page numbers
        
        shutil.copy2(self.build_folder + '/report.pdf', self.report_folder + '/report.pdf')
        
    @staticmethod
    def get_template(template_file):
        """define jinja environment compatible with latex"""
        
        latex_jinja_env = jinja2.Environment(
            block_start_string = '\BLOCK{',
            block_end_string = '}',
            variable_start_string = '\VAR{',
            variable_end_string = '}',
            comment_start_string = '\#{',
            comment_end_string = '}',
            line_statement_prefix = '%%',
            line_comment_prefix = '%#',
            trim_blocks = True,
            autoescape = False,
            loader = jinja2.FileSystemLoader(os.path.abspath('/'))
        ) # See http://akuederle.com/Automatization-with-Latex-and-Python-2 for more info
        template = latex_jinja_env.get_template(template_file)
        return template
        

def main(argv):
    """main function of cryptoledger."""
    
    # handle command line paramters
    parser = argparse.ArgumentParser() # initialize parser
    parser = argparse.ArgumentParser(description='Program description')
    parser.add_argument('ledger_db', type=str, help='path to the ledger *.csv file')
    parser.add_argument('-a','--add', metavar='csvs_folder_path', type=str, help="add transactions in all csv files in given folder to ledger")
    parser.add_argument('-m','--manual', action='store_true', help="manual transaction input") 
    parser.add_argument('-e','--export', action='store_true' , help='export pdf report')
    parser.add_argument('--currency', metavar='symbol', default='EUR', type=str, help="report currency (default: %(default)s)'")
    parser.add_argument('--start_date', metavar='date', default='first', type=str, help="report starting date formatted as YYYY-MM-DD (default: %(default)s)'")
    parser.add_argument('--end_date', metavar='date', default='last', type=str, help="report ending date fromatted as YYYY-MM-DD (default: %(default)s)'")
    args = parser.parse_args()
    
    # initiliaze ledge
    ledger = Ledger(args.ledger_db)
    
    # import transactions from csv
    if args.add is not None:
        print('importing *.csv from folder: ', os.path.abspath(args.add))
        for filename in os.listdir(os.path.abspath(args.add)):
            if filename.endswith('.csv'):
                file_path = args.add + '/' + filename
                print('Importing:', file_path)
                update_exchange_name = file_path.split('/')[-1].split('_')[0] # get the first word from the filename at the end of path
                ledger.append(file_path,update_exchange_name)
        ledger.save()
            
    # manually add a transaction
    if args.manual:
        ledger.manual_append()
        ledger.save()
        
    # print report
    if args.export is True:
        if ledger.transactions:
            portfolio = Portfolio(ledger, args.currency) # create portfolio and evaluate in currency passed in argument
            report = Report(portfolio, './report', start=args.start_date, end=args.end_date) # report latex template must be in ./reportfoled
            report.generate()
            report.compile_pdf()
            #~ plt.show()
        else:
            print('Ledger is empty')
    

if __name__ == "__main__":
    main(sys.argv)
