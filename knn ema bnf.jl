//@version=5
strategy('Machine Learning: kNN-based Strategy with EMA', 'ML-kNN-EMA', overlay=true, max_labels_count=300, format=format.price, precision=2)

// kNN-based Strategy (FX and Crypto)
// Description: 
// This strategy uses a classic machine learning algorithm - k Nearest Neighbours (kNN) - 
// to let you find a prediction for the next (tomorrow's, next month's, etc.) market move. 
// Being an unsupervised machine learning algorithm, kNN is one of the most simple learning algorithms. 

// To do a prediction of the next market move, the kNN algorithm uses the historic data, 
// collected in 3 arrays - feature1, feature2 and directions, - and finds the k-nearest 
// neighbours of the current indicator(s) values. 

// The two dimensional kNN algorithm just has a look on what has happened in the past when 
// the two indicators had a similar level. It then looks at the k nearest neighbours, 
// sees their state and thus classifies the current point.

// The kNN algorithm offers a framework to test all kinds of indicators easily to see if they 
// have got any *predictive value*. One can easily add cog, wpr and others.
// Note: TradingViews's playback feature helps to see this strategy in action.
// Warning: Signals ARE repainting.

// Style tags: Trend Following, Trend Analysis
// Asset class: Equities, Futures, ETFs, Currencies and Commodities
// Dataset: FX Minutes/Hours+++/Days

//-- Preset Dates
int startdate = timestamp('01 Jan 2000 00:00:00 GMT+10')
int stopdate  = timestamp('31 Dec 2025 23:45:00 GMT+10')

//-- Inputs
StartDate  = input.time  (startdate, 'Start Date')
StopDate   = input.time  (stopdate,  'Stop Date')
Indicator  = input.string('All',     'Indicator',   ['RSI','ROC','CCI','Volume','All'])
ShortWindow = input.int   (14,        'Short Period [1..n]', 1)
LongWindow = input.int   (35,        'Long Period [2..n]',  2)
BaseK      = input.int   (250,       'Base No. of Neighbours (K) [5..n]', 5)
Filter     = input.bool  (false,     'Volatility Filter')
Bars       = input.int   (300,       'Bar Threshold [2..5000]', 2, 5000)

//-- Constants
var int BUY   = 1
var int SELL  = -1
var int CLEAR = 0

var int k     = math.floor(math.sqrt(BaseK))  // k Value for kNN algo

//-- Variables
// Training data, normalized to the range of [0,...,100]
var array<float> feature1   = array.new_float(0)  // [0,...,100]
var array<float> feature2   = array.new_float(0)  //    ...
var array<int>   directions = array.new_int(0)    // [-1; +1]

// Result data
var array<int>   predictions = array.new_int(0)
var float        prediction  = 0.0
var array<int>   bars        = array.new<int>(1, 0) // array used as a container for inter-bar variables

// Signals
var int          signal      = CLEAR

//-- Functions
minimax(float x, int p, float min, float max) => 
    float hi = ta.highest(x, p), float lo = ta.lowest(x, p)
    (max - min) * (x - lo) / (hi - lo) + min

cAqua(int g) => g > 9 ? #0080FFff : g > 8 ? #0080FFe5 : g > 7 ? #0080FFcc : g > 6 ? #0080FFb2 : g > 5 ? #0080FF99 : g > 4 ? #0080FF7f : g > 3 ? #0080FF66 : g > 2 ? #0080FF4c : g > 1 ? #0080FF33 : #00C0FF19
cPink(int g) => g > 9 ? #FF0080ff : g > 8 ? #FF0080e5 : g > 7 ? #FF0080cc : g > 6 ? #FF0080b2 : g > 5 ? #FF008099 : g > 4 ? #FF00807f : g > 3 ? #FF008066 : g > 2 ? #FF00804c : g > 1 ? #FF008033 : #FF008019

inside_window(float start, float stop) =>  
    time >= start and time <= stop

//-- Logic
bool window = inside_window(StartDate, StopDate)

// Calculate 5-period EMA
ema5 = ta.ema(close, 5)

// 3 pairs of predictor indicators, long and short each
float rs = ta.rsi(close,   LongWindow),        rf = ta.rsi(close,   ShortWindow)
float cs = ta.cci(close,   LongWindow),        cf = ta.cci(close,   ShortWindow)
float os = ta.roc(close,   LongWindow),        of = ta.roc(close,   ShortWindow)
float vs = minimax(volume, LongWindow, 0, 99), vf = minimax(volume, ShortWindow, 0, 99)

// TOADD or TOTRYOUT:
//    ta.cmo(close, LongWindow), ta.cmo(close, ShortWindow)
//    ta.mfi(close, LongWindow), ta.mfi(close, ShortWindow)
//    ta.mom(close, LongWindow), ta.mom(close, ShortWindow)

float f1 = switch Indicator
    'RSI'    => rs 
    'CCI'    => cs 
    'ROC'    => os 
    'Volume' => vs 
    => math.avg(rs, cs, os, vs)

float f2 = switch Indicator
    'RSI'    => rf 
    'CCI'    => cf
    'ROC'    => of
    'Volume' => vf 
    => math.avg(rf, cf, of, vf)

// Classification data, what happens on the next bar
int class_label = int(math.sign(close[1] - close[0])) // eq. close[1]<close[0] ? SELL : close[1]>close[0] ? BUY : CLEAR

// Use particular training period
if window
    // Store everything in arrays. Features represent a square 100 x 100 matrix,
    // whose row-column intersections represent class labels, showing historic directions
    array.push(feature1, f1)
    array.push(feature2, f2)
    array.push(directions, class_label)

// Uncomment the following statement (if barstate.islast) and tab everything below
// between BOBlock and EOBlock marks to see just the recent several signals gradually 
// showing up, rather than all the preceding signals

//if barstate.islast   

//==BOBlock    

// Core logic of the algorithm
int   size    = array.size(directions)
float maxdist = -999.0
// Loop through the training arrays, getting distances and corresponding directions.
for i = 0 to size - 1
    // Calculate the euclidean distance of current point to all historic points,
    // here the metric used might as well be a manhattan distance or any other.
    float d = math.sqrt(math.pow(f1 - array.get(feature1, i), 2) + math.pow(f2 - array.get(feature2, i), 2))
    
    if d > maxdist
        maxdist := d
        if array.size(predictions) >= k
            array.shift(predictions)
        array.push(predictions, array.get(directions, i))
        
//==EOBlock    

// Note: in this setup there's no need for distances array (i.e. array.push(distances, d)),
//       but the drawback is that a sudden max value may shadow all the subsequent values.
// One of the ways to bypass this is to:
// 1) store d in distances array,
// 2) calculate newdirs = bubbleSort(distances, directions), and then 
// 3) take a slice with array.slice(newdirs) from the end
        
// Get the overall prediction of k nearest neighbours
prediction := array.sum(predictions)   

bool filter = Filter ? ta.atr(10) > ta.atr(40) : true // filter out by volatility or ex. ta.atr(1) > ta.atr(10)...

// Now that we got a prediction for the next market move, we need to make use of this prediction and 
// trade it. The returns then will show if everything works as predicted.
// Over here is a simple long/short interpretation of the prediction, 
// but of course one could also use the quality of the prediction (+5 or +1) in some sort of way,
// ex. for position sizing.

bool long  = (prediction > 0) and filter
bool short = (prediction < 0) and filter
bool clear = not(long or short)

// Condition for long trade: Previous candle below EMA
bool prev_candle_below_ema = (high[1] < ema5[1]) and (low[1] < ema5[1])

// New condition: Previous candle's entire body above its 50% level
float prev_candle_mid = (high[1] + low[1]) / 2
bool prev_candle_above_mid = (open[1] > prev_candle_mid) and (close[1] > prev_candle_mid)

if (long and prev_candle_below_ema and prev_candle_above_mid)
    strategy.entry("Long", strategy.long)

// Exit trade at the close of the entry candle
if strategy.opentrades > 0
    strategy.close("Long")

if array.get(bars, 0) == Bars    // stop by trade duration
    signal := CLEAR
    array.set(bars, 0, 0)
else
    array.set(bars, 0, array.get(bars, 0) + 1)

signal := long ? BUY : short ? CLEAR : clear ? CLEAR : nz(signal[1])

int  changed         = ta.change(signal)
bool startLongTrade  = changed and (signal == BUY)

if (startLongTrade)
    strategy.entry("Long", strategy.long)
else if (changed and (signal == CLEAR))
    strategy.close("Long")
