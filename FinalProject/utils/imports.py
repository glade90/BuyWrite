# Core imports
import os, sys, shutil
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import datetime, time
import seaborn as sns
from datetime import datetime
from collections import defaultdict

from scipy.stats import norm

# Basic settings
sns.set(style="whitegrid")

# Project Dates
START_DATE = '2015-01-01'
END_DATE = '2025-04-30'

# Risk-free rate assumption
RISK_FREE_RATE = 0.02  # 2% annualized
debug = False