from extensions import db
from models import EducationArticle
from datetime import datetime
from app import app

# Use the existing Flask app

# Stock basics articles
stock_basics_articles = [
    {
        "title": "What is a Stock?",
        "content": """
<h3>Understanding Stocks</h3>
<p>A stock (also known as equity) represents a share in the ownership of a company. When you purchase a company's stock, you're buying a small piece of that company, called a share.</p>

<h4>Key Features of Stocks</h4>
<ul>
    <li><strong>Ownership:</strong> Each share represents a small ownership stake in the company.</li>
    <li><strong>Limited Liability:</strong> Shareholders are not personally liable for the company's debts.</li>
    <li><strong>Voting Rights:</strong> Common shareholders typically have voting rights in corporate decisions.</li>
    <li><strong>Dividends:</strong> Some companies distribute a portion of their earnings to shareholders as dividends.</li>
    <li><strong>Capital Appreciation:</strong> Stocks can increase in value over time, allowing investors to sell shares at a profit.</li>
</ul>

<h4>Types of Stocks</h4>
<p><strong>Common Stock:</strong> Represents basic ownership, includes voting rights but has lower priority in dividend payments and asset claims.</p>
<p><strong>Preferred Stock:</strong> Has priority over common stock for dividend payments and asset claims, but typically doesn't include voting rights.</p>

<h4>Stock Markets</h4>
<p>Stocks are bought and sold on stock exchanges like the New York Stock Exchange (NYSE), NASDAQ, Bombay Stock Exchange (BSE), and National Stock Exchange of India (NSE). These exchanges provide a marketplace where buyers and sellers can trade shares according to market forces.</p>

<h4>Why Companies Issue Stock</h4>
<p>Companies issue stock to raise capital for various purposes, such as:</p>
<ul>
    <li>Funding expansion or new projects</li>
    <li>Paying off debt</li>
    <li>Acquiring other companies</li>
    <li>Research and development</li>
</ul>

<h4>How Stock Prices Are Determined</h4>
<p>Stock prices are determined by supply and demand in the market. Factors that influence stock prices include:</p>
<ul>
    <li>Company performance and financial health</li>
    <li>Industry trends</li>
    <li>Economic conditions</li>
    <li>Investor sentiment</li>
    <li>News and events</li>
</ul>
""",
        "author": "StockSage Team",
        "category": "basics",
        "featured": True
    },
    {
        "title": "Understanding Stock Market Indices",
        "content": """
<h3>Stock Market Indices Explained</h3>
<p>A stock market index is a measurement of a section of the stock market. It is computed from the prices of selected stocks, typically a weighted average. It helps investors compare current price levels with past prices to calculate market performance.</p>

<h4>Major Global Indices</h4>
<ul>
    <li><strong>S&P 500:</strong> Tracks the performance of 500 large companies listed on stock exchanges in the United States.</li>
    <li><strong>Dow Jones Industrial Average (DJIA):</strong> Price-weighted average of 30 significant stocks traded on the New York Stock Exchange and the NASDAQ.</li>
    <li><strong>NASDAQ Composite:</strong> Includes all companies listed on the NASDAQ stock market, with a heavy concentration of technology companies.</li>
    <li><strong>FTSE 100:</strong> Index of the 100 companies listed on the London Stock Exchange with the highest market capitalization.</li>
</ul>

<h4>Major Indian Indices</h4>
<ul>
    <li><strong>NIFTY 50:</strong> The National Stock Exchange of India's benchmark index for the Indian equity market, representing 50 of the largest Indian companies.</li>
    <li><strong>SENSEX:</strong> The S&P Bombay Stock Exchange Sensitive Index, comprising 30 of the largest and most actively traded stocks on the BSE.</li>
    <li><strong>NIFTY Bank:</strong> Index of the most liquid and large capitalized Indian banking stocks.</li>
    <li><strong>NIFTY IT:</strong> Index that tracks the performance of IT companies listed on the NSE.</li>
</ul>

<h4>How Indices Are Used</h4>
<p>Investors use market indices for several purposes:</p>
<ul>
    <li><strong>Benchmarking:</strong> Comparing investment performance against a relevant market index.</li>
    <li><strong>Market Sentiment:</strong> Gauging overall market direction and investor sentiment.</li>
    <li><strong>Passive Investing:</strong> Index funds and ETFs aim to replicate the performance of specific indices.</li>
    <li><strong>Economic Indicators:</strong> Indices often reflect broader economic conditions and trends.</li>
</ul>

<h4>Index Calculation Methods</h4>
<p>Indices can be calculated using different methods:</p>
<ul>
    <li><strong>Price-Weighted:</strong> Based on the stock prices of each company (e.g., Dow Jones Industrial Average).</li>
    <li><strong>Market Capitalization-Weighted:</strong> Based on the total market value of the constituent companies (e.g., S&P 500, NIFTY 50).</li>
    <li><strong>Equal-Weighted:</strong> Each stock has the same weight regardless of price or market cap.</li>
</ul>

<p>Understanding market indices helps investors track market performance and make informed investment decisions based on broader market trends.</p>
""",
        "author": "StockSage Team",
        "category": "basics",
        "featured": False
    },
    {
        "title": "How to Read Stock Charts",
        "content": """
<h3>The Basics of Stock Chart Analysis</h3>
<p>Stock charts are visual representations of a stock's price movements over time. Learning to read these charts is essential for technical analysis and making informed investment decisions.</p>

<h4>Types of Stock Charts</h4>
<ul>
    <li><strong>Line Charts:</strong> The simplest form, showing closing prices connected by a line.</li>
    <li><strong>Bar Charts:</strong> Display the open, high, low, and close (OHLC) prices for each period.</li>
    <li><strong>Candlestick Charts:</strong> Similar to bar charts but with "candles" that show the relationship between opening and closing prices.</li>
    <li><strong>Point and Figure Charts:</strong> Focus on price movements without regard to time.</li>
</ul>

<h4>Key Chart Components</h4>
<ul>
    <li><strong>Price Scale:</strong> Vertical axis showing the stock price.</li>
    <li><strong>Time Scale:</strong> Horizontal axis representing time periods (minutes, hours, days, weeks, months).</li>
    <li><strong>Volume:</strong> Bars at the bottom showing the number of shares traded in each period.</li>
    <li><strong>Moving Averages:</strong> Lines that show the average price over a specific number of periods.</li>
    <li><strong>Support and Resistance Levels:</strong> Price levels where a stock tends to stop falling or rising.</li>
</ul>

<h4>Common Chart Patterns</h4>
<p><strong>Trend Patterns:</strong></p>
<ul>
    <li><strong>Uptrend:</strong> Series of higher highs and higher lows.</li>
    <li><strong>Downtrend:</strong> Series of lower highs and lower lows.</li>
    <li><strong>Sideways/Horizontal:</strong> Price moves within a range with no clear direction.</li>
</ul>

<p><strong>Reversal Patterns:</strong></p>
<ul>
    <li><strong>Head and Shoulders:</strong> Indicates a potential trend reversal from bullish to bearish.</li>
    <li><strong>Double Top/Bottom:</strong> Shows resistance or support being tested twice before a reversal.</li>
    <li><strong>Cup and Handle:</strong> Bullish continuation pattern resembling a cup with a handle.</li>
</ul>

<p><strong>Continuation Patterns:</strong></p>
<ul>
    <li><strong>Flags and Pennants:</strong> Brief consolidation before continuing in the same direction.</li>
    <li><strong>Triangles:</strong> Converging trendlines indicating a potential breakout.</li>
    <li><strong>Rectangles:</strong> Price bounces between parallel support and resistance lines.</li>
</ul>

<h4>Technical Indicators</h4>
<p>Charts often include technical indicators that help analyze price movements:</p>
<ul>
    <li><strong>Relative Strength Index (RSI):</strong> Measures the speed and change of price movements.</li>
    <li><strong>Moving Average Convergence Divergence (MACD):</strong> Shows the relationship between two moving averages.</li>
    <li><strong>Bollinger Bands:</strong> Indicate volatility by showing price channels around a moving average.</li>
    <li><strong>Stochastic Oscillator:</strong> Compares a stock's closing price to its price range over a specific period.</li>
</ul>

<p>Learning to read stock charts takes practice, but it's a valuable skill for any investor looking to make more informed decisions based on technical analysis.</p>
""",
        "author": "StockSage Team",
        "category": "technical",
        "featured": False
    }
]

# Prediction model articles
prediction_model_articles = [
    {
        "title": "Understanding ARIMA Models for Stock Prediction",
        "content": """
<h3>ARIMA Models in Stock Market Prediction</h3>
<p>ARIMA (AutoRegressive Integrated Moving Average) is a popular statistical model used for time series forecasting, including stock price prediction. It combines three components: Autoregression (AR), Integration (I), and Moving Average (MA).</p>

<h4>Components of ARIMA</h4>
<ul>
    <li><strong>AutoRegressive (AR):</strong> Uses the dependent relationship between an observation and a number of lagged observations.</li>
    <li><strong>Integrated (I):</strong> Represents the differencing of observations to make the time series stationary.</li>
    <li><strong>Moving Average (MA):</strong> Uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.</li>
</ul>

<h4>ARIMA Parameters</h4>
<p>ARIMA models are denoted as ARIMA(p,d,q) where:</p>
<ul>
    <li><strong>p:</strong> The number of lag observations (lag order) in the model.</li>
    <li><strong>d:</strong> The number of times the raw observations are differenced (degree of differencing).</li>
    <li><strong>q:</strong> The size of the moving average window (order of moving average).</li>
</ul>

<h4>How ARIMA Works for Stock Prediction</h4>
<ol>
    <li><strong>Data Preparation:</strong> Historical stock price data is collected and prepared.</li>
    <li><strong>Stationarity Check:</strong> The time series must be stationary (constant mean, variance, and autocorrelation over time). If not, differencing is applied.</li>
    <li><strong>Parameter Selection:</strong> Optimal values for p, d, and q are determined using techniques like ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots.</li>
    <li><strong>Model Fitting:</strong> The ARIMA model is fitted to the historical data.</li>
    <li><strong>Forecasting:</strong> The model predicts future stock prices based on the patterns learned from historical data.</li>
    <li><strong>Evaluation:</strong> The model's accuracy is assessed using metrics like RMSE (Root Mean Square Error) or MAE (Mean Absolute Error).</li>
</ol>

<h4>Advantages of ARIMA</h4>
<ul>
    <li>Effective for short-term forecasting</li>
    <li>Captures linear relationships in time series data</li>
    <li>Well-established statistical foundation</li>
    <li>Relatively simple to implement and interpret</li>
</ul>

<h4>Limitations of ARIMA</h4>
<ul>
    <li>Assumes linear relationships between past and future values</li>
    <li>May not capture complex market dynamics</li>
    <li>Less effective for long-term forecasting</li>
    <li>Doesn't account for external factors affecting stock prices</li>
    <li>Requires stationary data</li>
</ul>

<h4>Extensions of ARIMA</h4>
<ul>
    <li><strong>SARIMA:</strong> Seasonal ARIMA, which includes seasonal components.</li>
    <li><strong>ARIMAX:</strong> ARIMA with exogenous variables, allowing external factors to be included.</li>
    <li><strong>GARCH:</strong> Often combined with ARIMA to model volatility in financial time series.</li>
</ul>

<p>ARIMA models remain a fundamental tool in quantitative finance, providing a statistical approach to understanding and predicting stock price movements based on historical patterns.</p>
""",
        "author": "StockSage Team",
        "category": "technical",
        "featured": True
    },
    {
        "title": "LSTM Networks for Stock Price Prediction",
        "content": """
<h3>Long Short-Term Memory (LSTM) Networks in Stock Market Prediction</h3>
<p>LSTM is a type of recurrent neural network (RNN) architecture designed to recognize patterns in sequences of data, such as time series of stock prices. Unlike traditional RNNs, LSTMs are capable of learning long-term dependencies, making them particularly useful for stock market prediction.</p>

<h4>How LSTM Networks Work</h4>
<p>LSTM networks contain special units called memory cells that can maintain information for long periods. Each memory cell contains:</p>
<ul>
    <li><strong>Input Gate:</strong> Controls when new information flows into the cell.</li>
    <li><strong>Forget Gate:</strong> Controls when information is removed from the cell.</li>
    <li><strong>Output Gate:</strong> Controls when information in the cell is used in the output.</li>
    <li><strong>Cell State:</strong> Carries information throughout the processing of the sequence.</li>
</ul>

<h4>LSTM for Stock Price Prediction</h4>
<ol>
    <li><strong>Data Preparation:</strong> Historical stock data is collected, normalized, and divided into training and testing sets.</li>
    <li><strong>Sequence Creation:</strong> Data is organized into sequences where each sequence represents a window of past stock prices.</li>
    <li><strong>Model Architecture:</strong> An LSTM network is designed with appropriate layers and neurons.</li>
    <li><strong>Training:</strong> The model learns patterns from historical data by adjusting its weights through backpropagation.</li>
    <li><strong>Prediction:</strong> The trained model forecasts future stock prices based on recent price sequences.</li>
    <li><strong>Evaluation:</strong> The model's performance is assessed using metrics like RMSE or MAE.</li>
</ol>

<h4>Advantages of LSTM for Stock Prediction</h4>
<ul>
    <li>Ability to capture long-term dependencies in time series data</li>
    <li>Can learn complex non-linear relationships</li>
    <li>Robust to noise and missing values in the data</li>
    <li>Can process multivariate inputs (price, volume, technical indicators)</li>
    <li>Adaptable to changing market conditions through retraining</li>
</ul>

<h4>Limitations of LSTM</h4>
<ul>
    <li>Requires large amounts of data for effective training</li>
    <li>Computationally intensive and may require significant resources</li>
    <li>Prone to overfitting without proper regularization</li>
    <li>Difficult to interpret the learned patterns (black box nature)</li>
    <li>May struggle with extreme market events not represented in training data</li>
</ul>

<h4>LSTM Variants and Enhancements</h4>
<ul>
    <li><strong>Bidirectional LSTM:</strong> Processes sequences in both forward and backward directions.</li>
    <li><strong>Stacked LSTM:</strong> Uses multiple LSTM layers for more complex pattern recognition.</li>
    <li><strong>Attention Mechanisms:</strong> Helps the model focus on relevant parts of the input sequence.</li>
    <li><strong>CNN-LSTM Hybrid:</strong> Combines convolutional neural networks with LSTM for feature extraction and sequence learning.</li>
</ul>

<h4>Feature Engineering for LSTM Models</h4>
<p>Effective LSTM models often incorporate various features beyond just price data:</p>
<ul>
    <li>Technical indicators (RSI, MACD, Bollinger Bands)</li>
    <li>Trading volume</li>
    <li>Market sentiment from news and social media</li>
    <li>Macroeconomic indicators</li>
    <li>Cross-market correlations</li>
</ul>

<p>LSTM networks represent a powerful deep learning approach to stock market prediction, capable of capturing complex temporal patterns that traditional statistical methods might miss.</p>
""",
        "author": "StockSage Team",
        "category": "technical",
        "featured": True
    },
    {
        "title": "Linear Regression in Stock Market Analysis",
        "content": """
<h3>Linear Regression for Stock Price Prediction</h3>
<p>Linear regression is one of the simplest yet effective statistical methods used in stock market analysis. It models the relationship between a dependent variable (stock price) and one or more independent variables (time, volume, economic indicators, etc.).</p>

<h4>Basic Concept of Linear Regression</h4>
<p>Linear regression attempts to model the relationship by fitting a linear equation to the observed data:</p>
<p><strong>Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε</strong></p>
<p>Where:</p>
<ul>
    <li>Y is the dependent variable (stock price or return)</li>
    <li>X₁, X₂, ..., Xₙ are independent variables (predictors)</li>
    <li>β₀, β₁, β₂, ..., βₙ are the coefficients</li>
    <li>ε is the error term</li>
</ul>

<h4>Types of Linear Regression in Stock Analysis</h4>
<ul>
    <li><strong>Simple Linear Regression:</strong> Uses a single predictor variable, such as time.</li>
    <li><strong>Multiple Linear Regression:</strong> Uses multiple predictor variables, such as volume, market indices, and economic indicators.</li>
    <li><strong>Polynomial Regression:</strong> Fits a non-linear relationship using polynomial terms.</li>
</ul>

<h4>Applications in Stock Market Analysis</h4>
<ol>
    <li><strong>Trend Analysis:</strong> Identifying the direction and strength of a stock's price movement over time.</li>
    <li><strong>Factor Models:</strong> Analyzing how various factors affect stock returns (e.g., Fama-French Three-Factor Model).</li>
    <li><strong>Beta Calculation:</strong> Measuring a stock's volatility relative to the market.</li>
    <li><strong>Price Forecasting:</strong> Predicting future stock prices based on historical data and other variables.</li>
    <li><strong>Valuation:</strong> Determining if a stock is overvalued or undervalued based on its relationship with fundamental factors.</li>
</ol>

<h4>Implementing Linear Regression for Stock Prediction</h4>
<ol>
    <li><strong>Data Collection:</strong> Gather historical stock prices and potential predictor variables.</li>
    <li><strong>Data Preprocessing:</strong> Clean the data, handle missing values, and normalize if necessary.</li>
    <li><strong>Feature Selection:</strong> Identify which variables have significant predictive power.</li>
    <li><strong>Model Training:</strong> Fit the linear regression model to the historical data.</li>
    <li><strong>Model Evaluation:</strong> Assess the model's performance using metrics like R-squared, RMSE, and p-values.</li>
    <li><strong>Prediction:</strong> Use the model to forecast future stock prices or returns.</li>
</ol>

<h4>Advantages of Linear Regression</h4>
<ul>
    <li>Simple to understand and implement</li>
    <li>Computationally efficient</li>
    <li>Provides interpretable coefficients that explain the relationship between variables</li>
    <li>Works well for identifying linear relationships in data</li>
    <li>Serves as a baseline for more complex models</li>
</ul>

<h4>Limitations of Linear Regression</h4>
<ul>
    <li>Assumes a linear relationship between variables</li>
    <li>Sensitive to outliers</li>
    <li>May not capture complex, non-linear market dynamics</li>
    <li>Assumes independence of observations (which may not hold for time series data)</li>
    <li>Limited predictive power in highly volatile markets</li>
</ul>

<h4>Enhancements to Basic Linear Regression</h4>
<ul>
    <li><strong>Ridge Regression:</strong> Adds a penalty term to reduce overfitting.</li>
    <li><strong>Lasso Regression:</strong> Performs feature selection by shrinking some coefficients to zero.</li>
    <li><strong>Elastic Net:</strong> Combines ridge and lasso penalties for better performance.</li>
    <li><strong>Quantile Regression:</strong> Models different parts of the distribution, not just the mean.</li>
</ul>

<p>While more sophisticated models like LSTM and ARIMA often outperform linear regression for stock prediction, it remains a valuable tool for understanding relationships between variables and providing a benchmark for more complex approaches.</p>
""",
        "author": "StockSage Team",
        "category": "technical",
        "featured": False
    }
]

# Function to add articles to the database
def add_articles():
    with app.app_context():
        # Add stock basics articles
        for article_data in stock_basics_articles:
            # Check if article with same title already exists
            existing_article = EducationArticle.query.filter_by(title=article_data['title']).first()
            if not existing_article:
                article = EducationArticle(
                    title=article_data['title'],
                    content=article_data['content'],
                    author=article_data['author'],
                    category=article_data['category'],
                    featured=article_data['featured'],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.session.add(article)
                print(f"Added article: {article_data['title']}")
            else:
                print(f"Article already exists: {article_data['title']}")
        
        # Add prediction model articles
        for article_data in prediction_model_articles:
            # Check if article with same title already exists
            existing_article = EducationArticle.query.filter_by(title=article_data['title']).first()
            if not existing_article:
                article = EducationArticle(
                    title=article_data['title'],
                    content=article_data['content'],
                    author=article_data['author'],
                    category=article_data['category'],
                    featured=article_data['featured'],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.session.add(article)
                print(f"Added article: {article_data['title']}")
            else:
                print(f"Article already exists: {article_data['title']}")
        
        # Commit all changes
        db.session.commit()
        print("All articles have been added successfully!")

if __name__ == "__main__":
    add_articles()
