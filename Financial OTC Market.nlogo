extensions [ py ]

globals [ a returns p0 p1 price-history histogram-num-bars obs-length bidofferpaid ai-trained-str]

breed [ valueinvestors valueinvestor ]
breed [ smartinvestors smartinvestor ]
breed [ dealers dealer ]

turtles-own [ expectation inventory ]
links-own [ weight ] ; weight will represent the size of trade between two parties for visual illustration
dealers-own [ bid offer last-trade ] ; bid and offer prices
valueinvestors-own [ uncertainty ]
smartinvestors-own [ confident? trade-holding-times state-memory positions transaction-prices ctransaction-price recent-price-history actions_index ]


to setup ; global procedure

  clear-all

  set obs-length 512 ; length of price history used to train the smart valueinvestors RL. Keep at 500.
  set a 0.00001 ; constant determining how much bid-offer increases as size increases. Model parameter

  setup-python
  initialise-dealers
  initialise-valueinvestors
  initialise-smartinvestors

  ask links [ ; three boundary criteria: a turtle cannot have less than one connection to another dealer
    if weight > prob-of-link and count ( [ link-neighbors with [ breed = dealers ] ] of end2  ) > 1 and count ( [ link-neighbors with [ breed = dealers ] ] of end1 ) > 1 [
      die
    ]
    set color grey - 3
  ]

  set p0 price-level ; cache the price level
  set returns [] ; list for recording the returns
  set price-history []
  set ai-trained-str "A.I. training..."
  reset-ticks
end


to setup-python
  let path-to-python "/Users/jameswilkinson/opt/miniforge3/envs/Multi-Agent-RL-Market/bin/python"
  ; basic checks and error messages
  if path-to-python = "" [
    error "Path to python executable not set. Please update path-to-python to your python executable with the installed requirements outlined in the info-tab."
  ]

  if not file-exists? path-to-python [
    error "Python executable not found at end of path-to-python. Please update path-to-python to your python executable with the installed requirements outlined in the info-tab."
  ]

  py:setup path-to-python ; set up python
  py:run "import deepQlearn as q"
  py:run "agents = {}"
end


to go ; global procedure

  ; 1) get an investor at random to act
  ask one-of turtles [
    if breed = valueinvestors [
      valueinvestor-act
    ]
    if breed = smartinvestors [
      smartinvestor-act
    ]
    if ( breed = dealers ) and ( abs ( inventory ) > dealer-position-limit ) and enable-broker-market? [ ;
      dealer-act
    ]
  ]

  ; 2) refresh prices
  step
  record-returns
  update-graphics
  update-pricehistory

  tick

end


to update-pricehistory ; global procedure
  set price-history lput mean [ expectation ] of dealers price-history
  ask smartinvestors [
    set recent-price-history lput ( mean [ expectation ] of link-neighbors with [ breed = dealers ] - 100 ) recent-price-history ; -100 to make neural network inputs closer to 0
    if length recent-price-history > obs-length [
      set recent-price-history remove-item 0 recent-price-history ; keep the recent price history a fixed length
    ]
  ]
end


to update-graphics ; global procedure
    ask links [
      ifelse thickness < 0.2 [
        set color grey - 3
        set thickness 0
        set hidden? false
      ][
        set thickness thickness / 1.02
        set color color - 0.03
      ]
    ]
    ask dealers [
      if inventory > dealer-position-limit [
        set color green
      ]
      if inventory < -1 * dealer-position-limit [
        set color red
      ]
      if abs ( inventory ) < dealer-position-limit [
        set color grey + 3
      ]
    ]

  ; set the ai-train-str to update if the AI has trained yet or not
  ifelse not any? smartinvestors with [ not confident? ] [
    set ai-trained-str "A.I. live!"
  ][
    let eps_mean py:runresult "sum([agents[a].epsilon for a in agents])/len(agents)"
    let eps_min py:runresult "sum([agents[a].confidence_epsilon_thresh for a in agents])/len(agents)"
    let eps_dec py:runresult "sum([agents[a].epsilon_decay for a in agents])/len(agents)"
    let rate -1 * log (1 - eps_dec ) 2.71
    let pct_complete  ( exp ( ( ( 1 - eps_mean ) / ( 1 - eps_min ) ) - 1 ) * 100  - 37 ) / ( 1 - 0.37 )
    set ai-trained-str ( word "A.I. Training " ( max list 0 floor pct_complete ) "%" )
  ]

end


to record-returns ; global procedure
    if ( ticks mod 50 ) = 0 and not any? smartinvestors with [ not confident? ] [ ; record the price changes every 50 ticks
    set p1 price-level
    set returns lput (p1 - p0) returns
    set p0 p1
  ]
end


to step ; global procedure
  ask dealers [
    refresh-bidoffer
  ]

  let bestbid max [ bid ] of dealers
  let bestoffer min [ offer ] of dealers
end


to refresh-bidoffer ; dealer procedure
  let axe-adj a * sensitivity-function ( -1 * inventory  )
  set expectation last-trade + axe-adj

  set offer ( expectation + bid-offer / 2 )
  set bid ( expectation - bid-offer / 2 )
end


to dealer-act ; dealer procedure
  let trade_size min list ( trade-size-cap ) ( abs ( inventory ) )
  if inventory < -1 * dealer-position-limit [
    let bestdealer max-one-of link-neighbors with [ breed = dealers ] [ inventory ]
    ;set trade_size min list ( trade_size ) ( [ inventory ] of bestdealer ) ; dealers will only trade to make themselves flatter
    transact "buy" trade_size [ who ] of bestdealer false
  ]
  if inventory > dealer-position-limit [
    let bestdealer min-one-of link-neighbors with [ breed = dealers ] [ inventory ]
    ;set trade_size min list ( trade_size ) ( -1 * [ inventory ] of bestdealer ) ; dealers will only trade to make themselves flatter
    transact "sell" trade_size [ who ] of bestdealer false
  ]
end


to smartinvestor-act ; smart-investor procedure

  py:set "id" who

  if ( py:runresult "agents[id].confident" ) [
    set confident? true
  ]

  ifelse length trade-holding-times > 0 and min trade-holding-times <= ticks [ ; if a trade has ended, close out deterministically and remember + learn whether the action from the state went well
    let index position ( min trade-holding-times ) trade-holding-times
    let action_index item index actions_index
    let ptransaction-price ( item index transaction-prices )
    let _position ( item index positions )
    if _position > 0 [ ; if position > 0, then trade we're closing was a "buy". so we need to sell to reverse out.
      let bestdealer [ who ] of min-one-of link-neighbors with [ breed = dealers ] [ inventory ]
      transact "sell" abs ( _position ) bestdealer false
    ]
    if _position < 0 [
      let bestdealer [ who ] of max-one-of link-neighbors with [ breed = dealers ] [ inventory ]
      transact "buy" abs( _position ) bestdealer false
    ]

    set bidofferpaid bidofferpaid + bid-offer

    py:set "state" ( item index state-memory )
    py:set "next_state" recent-price-history
    let reward _position * ( ctransaction-price - ptransaction-price ) / ( ptransaction-price )
    py:set "reward" reward
    py:set "action" action_index
    if length ( item index state-memory ) = 512 [
      py:run "agents[id].remember(state, action, next_state, reward)"
    ]

    ; finally, remove storage of the trade which is now closed
    set positions remove-item index positions
    set trade-holding-times remove-item index trade-holding-times
    set transaction-prices remove-item index transaction-prices
    set state-memory remove-item index state-memory
    set actions_index remove-item index actions_index

  ][ ; else, make a new action

    py:set "state" recent-price-history
    let action py:runresult "agents[id].act(state)"
    py:set "action" action
    let act_list py:runresult "agents[id].get_action_details(action)"
    let buy_sell item 0 act_list
    let trade_size item 1 act_list
    let trade_time item 2 act_list

    set state-memory lput recent-price-history state-memory ; save the state associated with the action
    set trade-holding-times lput ( ticks + trade_time ) trade-holding-times ; add a new counter to the list

    set actions_index lput action actions_index
    if buy_sell = "buy" [
      let bestdealer [ who ] of max-one-of link-neighbors with [ breed = dealers ] [ inventory ]
      transact "buy" trade_size bestdealer true
    ]
    if buy_sell = "sell" [
      let bestdealer [ who ] of min-one-of link-neighbors with [ breed = dealers ] [ inventory ]
        transact "sell" trade_size bestdealer true
    ]
    if buy_sell = "do nothing" [
      set transaction-prices lput 100 transaction-prices ; this is a "do nothing" transaction, reward will be zero so transaction_price doesn't matter
      set positions lput 0 positions
    ]
  ]
end


to valueinvestor-act ; value-investor procedure

  let best_advertised_bid [ bid ] of min-one-of ( link-neighbors with [ breed = dealers ] ) [ inventory ] ; best bid in the market
  let best_advertised_offer [ offer ] of max-one-of ( link-neighbors with [ breed = dealers ] ) [ inventory ] ; best offer in the market

  let gain-from-buy ( expectation - best_advertised_offer )
  let gain-from-sell ( best_advertised_bid - expectation )

  ;; 1) Sell
  if ( best_advertised_bid > expectation ) and ( gain-from-buy < gain-from-sell ) [ ; if the price is real, and higher than expectation, sell
    let bestdealer max-one-of link-neighbors with [ breed = dealers ] [ -1 * inventory ] ; naturally, this will select the dealer with most room
    let bestbid [ bid ] of bestdealer
    ; if valueinvestoris not selling because of inventory limits, then act as normal. Else, trade to get within limits again
    let tradesize min list ( trade-size-cap ) ( ( bestbid - expectation ) / 5 )
    transact "sell" tradesize [ who ] of bestdealer false
  ]

   ;; 2) Buy
  if ( best_advertised_offer < expectation ) and ( gain-from-sell < gain-from-buy ) [ ; if the price is real, and lower than expectation, buy
    let bestdealer max-one-of link-neighbors with [ breed = dealers ] [ inventory ] ; naturally, this will select the dealer with most room
    let bestoffer [ offer ] of bestdealer
    let tradesize min list ( trade-size-cap ) ( ( expectation - bestoffer ) / 5 )
    transact "buy" tradesize [ who ] of bestdealer false
  ]

end


to transact [ direction tradesize counterpartyID record ] ; turtle procedure

  set tradesize abs ( tradesize )
  let factor 1 ;initialise
  let counterparty 0 ; initialise
  ifelse direction = "buy" [
    set factor 1
    set counterparty one-of dealers with [ inventory = max [ inventory ] of dealers ]
  ][
    set factor -1
    set counterparty one-of dealers with [ inventory = min [ inventory ] of dealers ]
  ]

  let bestpx 0 ; initialise
  let max-natural-size 0 ; initialise
  ifelse direction = "sell" [
    set bestpx [ bid ] of counterparty
  ][
    set bestpx [ offer ] of counterparty
  ]

  let final-px 100 ; initialise
  let final-size 0 ; initialise
  ifelse ( breed = valueinvestors ) [
    set final-px bestpx - a * ( [ inventory + expectation / 5 ] of one-of dealers with [ who = counterpartyID ] ) * 1. / ( 1 +  a / 5 )
    set final-size abs ( 0.2 * ( expectation - final-px ) )
  ][
    set final-px bestpx - a * [ inventory + ( factor * tradesize ) ] of one-of dealers with [ who = counterpartyID ]
    set final-size abs ( tradesize )
]
  ;let final-px bestpx - sensitivity-function ( [ inventory - factor * tradesize ] of counterparty - tradesize )
  ;let final-size tradesize

  if ( breed = smartinvestors ) [ ; NB: this would be nicer inside smartinvestor-act procedure, but it requires knowing size-adj-px so this is not an option.
    ifelse length trade-holding-times > 0 and min trade-holding-times <= ticks [ ; if it was an old trade being closed, remove the trade stats from the traders memory
      set ctransaction-price final-px
      ][ ; else create a new part in the memory
      if record [
        set positions lput ( factor * final-size ) positions
        set transaction-prices lput final-px transaction-prices
      ];
    ]
  ]

  set inventory inventory + factor * final-size
  ask dealers with [ in-link-neighbor? counterparty ] [
    set last-trade final-px
  ]
  ask counterparty [
    set last-trade final-px
    set inventory inventory - factor * final-size
  ]

  ask my-links with [ end1 = counterparty ] [
    set thickness min list 2 tradesize * 0.2
    set hidden? false
    ifelse direction = "sell" [
      set color red
    ][
      set color green
    ]
  ]

end


to initialise-smartinvestors ; global procedure
  let counter 0
  create-ordered-smartinvestors n-smart-investors [
    set shape "person"
    set size 2
    fd 16
    rt 180
    set color red
    create-links-with dealers [ set weight random 100 ]
    set inventory 0
    set state-memory []

    py:set "params" []
    py:set "id" who
    py:set "lr" 1e-3
    py:set "state_size" obs-length
    py:set "eps_decay" 0.995
    py:run "agents[id] = q.SmartTrader(lr, state_size, eps_decay=eps_decay, batch_size=128)"

    set trade-holding-times []
    set positions []
    set transaction-prices []
    set recent-price-history []
    set actions_index []
    set confident? false
  ]
end


to initialise-valueinvestors; global procedure
  create-ordered-valueinvestors n-value-investors [
    set shape "person"
    set size 2
    fd 13
    rt 180
    set color blue
    set uncertainty 5 ; allow value investor expectations to be drawn from a gaussian with stdev 5
    ifelse random 100 < 50 [
      set expectation random-normal ( 100 - market-disparity ) ( uncertainty )
    ][
      set expectation random-normal ( 100 + market-disparity ) ( uncertainty )
    ]
    create-links-with dealers [ set weight random 100 ]
    set inventory 0
  ]
end


to initialise-dealers ; global procedure
  create-ordered-dealers n-dealers [
    set shape "house"
    set size 3
    fd 5
    rt 180
    set color grey
    set expectation 100
    set bid expectation - bid-offer / 2
    set offer expectation + bid-offer / 2
    create-links-with other dealers [ set weight random 100 ]
    set inventory 0
    set last-trade 100 ; initialise at the starting price
  ]
end


to plot-normal [ histogram-bins ] ; global procedure
  if not any? smartinvestors with [ not confident? ] [
    let meanPlot mean returns
    let varPlot variance returns
    let areascaler length returns * ( plot-x-max - plot-x-min ) / histogram-bins
    let stepper plot-x-min
    let mult areascaler / sqrt ( 2 * pi * varPlot )
    plot-pen-reset
    while [ stepper < plot-x-max ]
    [
      plotxy stepper (mult * exp ( - ((stepper - meanPlot) ^ 2) / (2 * varPlot) ) )
      set stepper stepper + 0.01
    ]
  ]
end

to-report kurtosis
  ifelse length returns > 20 [
    let m4 sum ( map [ i -> ( i - mean returns ) ^ 4 ] returns ) / ( length returns )
    let m2 sum ( map [ i -> ( i - mean returns ) ^ 2 ] returns ) / ( length returns )
    report m4 / m2 ^ 2
  ][
    report 0
  ]
end

to-report skew
  ifelse length returns > 20 [
    let m3 sum ( map [ i -> ( i - mean returns ) ^ 3 ] returns ) / ( length returns )
    let m2 sum ( map [ i -> ( i - mean returns ) ^ 2 ] returns ) / ( length returns )
    report m3 / m2 ^ 2
  ][
    report 0
  ]
end

to move-market ; global procedure
  ask valueinvestors [
    set expectation expectation * 0.8 ; force a market crash by reducing the value investor expectation
  ]
end

to force-dealers-short ; global procedure. when called, forces dealers to get limit-long
  ask dealers [
    set inventory -1 * dealer-position-limit * 1.5
  ]
end

to kill-value-investors
  ask valueinvestors [
    die
  ]
end

to-report normal [x mu var]
  report 1 / sqrt ( 2 * pi * var ) * exp ( - ((x - mu) ^ 2) / (2 * var) )
end

to-report price-level
  report mean [ expectation ] of dealers
end

to-report best-bid
  let bestbid [ bid ] of max-one-of dealers [ bid ]
  ifelse bestbid > -1e10 [
    report bestbid
  ][
    report "No Bid"
  ]
end

to-report best-offer
  let bestoffer [ offer ] of min-one-of dealers [ offer ]
  ifelse bestoffer < 1e10 [
    report bestoffer
  ][
    report "No Offer"
  ]
end

to-report rollingVol [ period return_period ]
  ;if length returns >= period [
  ;  let vol2 variance ( sublist returns ( length returns - period ) ( length returns ) )
  ;  report sqrt vol2
  ;]
  if length price-history >= return_period * period [
    let differences []
    let counter 0
    while [ counter < period ] [
      set differences lput ( ( item ( (length price-history) - return_period - period + counter ) price-history ) - ( item ( length price-history - period + counter - 1 ) price-history ) ) differences
      set counter counter + 1
    ]
    report sqrt variance differences
  ]

end

to-report sensitivity-function [ x ]
  report a * x  ; linear
end

to-report smartinvestor-rewards
  let meanreward ( py:runresult "sum([agents[id].totalreward for id in agents.keys()])" ) / n-smart-investors
  report meanreward
end

to-report smartinvestor-epsilon
  let epsilon py:runresult "sum([agents[id].epsilon for id in agents.keys()])" / n-smart-investors
  report epsilon
end

to-report z-score [x _mean var ]; z scores x on normal distro with _mean and var
  report abs( x - _mean ) / sqrt ( var )
end

to update-price-plot; plot procedure
  set-plot-x-range ( max list ( 0 ) ( ticks - 2000 ) ) ticks + 10
  if length price-history > 2000 [
  let last2000 sublist price-history ( length price-history - 2000 ) ( length price-history )
  let ymax max last2000
  let ymin min last2000
  set-plot-y-range ( precision ymin 2 ) ( precision ymax 2 )
  ]
end

to update-histogram; plot procedure
  clear-plot
  set-plot-x-range ( -1 * bid-offer * 20 ) ( bid-offer * 20 )
  set histogram-num-bars 2 * ( plot-x-max - plot-x-min ) / bid-offer
end
@#$#@#$#@
GRAPHICS-WINDOW
776
346
1150
721
-1
-1
11.1
1
10
1
1
1
0
0
0
1
-16
16
-16
16
0
0
1
ticks
30.0

SLIDER
17
144
189
177
n-value-investors
n-value-investors
0
40
10.0
1
1
NIL
HORIZONTAL

BUTTON
28
10
94
43
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

MONITOR
427
265
494
310
live offer
best-offer
2
1
11

MONITOR
353
264
418
309
live bid
best-bid
2
1
11

BUTTON
111
52
180
85
go
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
28
52
93
85
go-once
go
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

PLOT
236
315
736
523
Price
ticks
price
0.0
10.0
0.0
10.0
true
true
"set-plot-y-range 99 101\nset-plot-x-range 0 100" "update-price-plot"
PENS
"market bid" 1.0 0 -13345367 true "" "let bestbid [ bid ] of max-one-of dealers [ bid ]\nlet bestoffer [ offer ] of min-one-of dealers [ offer ]\nifelse bestbid = -1e11 [ \nset-plot-pen-color white\nplot bestoffer - bid-offer\n][\nset-plot-pen-color blue\nplot bestbid\n]"
"market offer" 1.0 0 -2674135 true "" "let bestoffer [ offer ] of min-one-of dealers [ offer ]\nlet bestbid [ bid ] of max-one-of dealers [ bid ]\nifelse bestoffer > 1e10 [\nset-plot-pen-color white \nplot bestbid + bid-offer\n][\nset-plot-pen-color red\nplot bestoffer\n]"

SLIDER
16
278
188
311
bid-offer
bid-offer
0
2
0.7
0.1
1
NIL
HORIZONTAL

SLIDER
16
326
188
359
dealer-position-limit
dealer-position-limit
0
100
50.0
1
1
NIL
HORIZONTAL

PLOT
459
72
740
239
Realised Return
50 tick change
Count
-10.0
10.0
0.0
10.0
true
true
"update-histogram" ""
PENS
"Observed" 1.0 1 -16777216 true "set-histogram-num-bars histogram-num-bars" "set-histogram-num-bars histogram-num-bars\nhistogram returns"
"Normal" 1.0 2 -5298144 true "" "plot-normal histogram-num-bars"

SLIDER
19
187
191
220
n-dealers
n-dealers
2
20
10.0
1
1
NIL
HORIZONTAL

BUTTON
263
153
377
186
Market Crash
move-market
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
236
204
403
237
Force Dealers short
Force-Dealers-Short
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
16
416
188
449
trade-size-cap
trade-size-cap
0
50
3.0
1
1
NIL
HORIZONTAL

SLIDER
19
233
191
266
n-smart-investors
n-smart-investors
0
30
5.0
1
1
NIL
HORIZONTAL

SLIDER
16
369
188
402
prob-of-link
prob-of-link
0
100
100.0
1
1
NIL
HORIZONTAL

SLIDER
16
462
188
495
market-disparity
market-disparity
0
20
20.0
1
1
NIL
HORIZONTAL

BUTTON
244
102
400
135
Remove Value Investors
kill-value-investors
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

MONITOR
671
129
736
174
Kurtosis
kurtosis
3
1
11

MONITOR
670
179
737
224
Skew
skew
3
1
11

BUTTON
111
10
182
43
repeat
ifelse ticks > 30000 [ \n  setup\n  go\n][\n  go\n]
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SWITCH
15
100
190
133
enable-broker-market?
enable-broker-market?
0
1
-1000

MONITOR
548
264
665
309
A.I. status
ai-trained-str
17
1
11

TEXTBOX
256
73
406
91
Market Scenarios to try:
11
0.0
1

TEXTBOX
507
32
689
59
Wait for the A.I. to train to see resulting price distribution chart.
11
0.0
1

PLOT
777
20
1351
285
smart-investor profit
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot smartinvestor-rewards"

@#$#@#$#@
## WHAT IS IT?

This model simulates a price within a financial market. Three types of agent - value investors, reinforcement-learning investors, and market makers (dealers) - trade against one another, causing the market price to change. The price movements produced by this model resemble those seen in the real world, with fat tailed distributions, volatility clustering and technicals.


## HOW IT WORKS

This market comprises of a single product, that agents can either buy, or sell. Investors actively buy and sell this product based on their opinion about its future price. Some investors, whom we call value investors, have static, long-term opinions about the future price. Other investors, which we call smart investors, have dynamic, shorter-term opinions aimed at exploiting forseable behaviour within the system. Dealers are counterparties to all trades and will move their price according to both the trades they see happening, and their inventory.

Value-investors are initialised with a randomly generated number in the vicinity of the inital market price. If their number is higher than the market price, they will choose to buy from a dealer (and vice versa if it is lower). The value-investor is a direct analogy to investors using fundamental analysis.

Smart-investors use artificial intelligence to maximize their trading profit. These agents make decisions informed by the market price history. Smart-investors are a direct analogy to investors using technical analysis, and their behaviour can encompass trend-following strategies, mean-reversion strategies, and more.

## THINGS TO NOTICE

Clicking the go button within the view will begin the model. Initially, notice the price chart updating, separately showing the optimal bid and offer price across all of the dealers at each time step.

The user will have to wait until the A.I. is fully trained before price movements are illustrated on the histrogram, which will be indicated by the A.I. Status monitor in the view. Notice if the price changes is normally distributed or not. The kurtosis of the distribution is displayed with the histrogram, and a kurtosis above 3 implies a fat-tailed distribution.

Notice that configurations with relatively few dealers result in fat-tailed price changes. Conversely, increasing the number of dealers results in more normally-distributed price changes.

Notice how the commission within the market (the difference between the most competitive offer price and the most competitive bid price) varies over time, and how it is always below the fixed bid offer.

## Set-up Instructions

This model uses NetLogo's python extension to manage the deep Q-learning framework. This must be set up properly. 

Firstly, a path to the preferred python executable must be specified by the user within the view's textbox labelled "path-to-python".

Secondly, this python executable must already have the requirements installed that are outlined below:

numpy==1.21.6
torch==1.11.0


## HOW TO USE IT

You can intuitively adjust the structure of the market, varying the number of dealers, value investors, smart investors, and more.

### Slider functions

n-value-investors : the number of value-investors (blue person turtles) initialised in the model

n-dealers : the number of dealers (house turtles) initialised in the model

n-smart-investors : the number of A.I. investors (red person turtles) initialised in the model

bid-offer : the commission associated with one trade. This is the difference between a dealer's offer (sell) price and bid (buy) price.

dealer-position-limit : the position size a market maker can accumulate before they begin skewwing their price, or (if the mm-limit-toggle is turned on) before a market-maker will transact with other market-makers to reduce their position size.

prob-of-link : the probability of a link forming between a turtle and a market maker. NB: This is constrained so every turtle is connected to at least one market maker.

trade-size-cap : the limit to the size of a single trade allowed to be executed by a value investor, and (if the mm-limit-toggle is turned on) between market makers.

market-disparity : the difference between the two-normal distribution means from which value-investor opinions are drawn from. When the market-disparity is increased, the two normal distributions become centered equidistant on either side of 100 by this amount. When market-disparity is zero, all value investors will have their expectation drawn from a normal distribution centered at 100. If the market disparity is 20, approximately half of the value-investors have their expectations drawn from a normal distribution centered at 120, whilst the other half have their normal distribution centered at 80.


### Toggles

enable-broker-market : turn ON to trigger dealer behaviour, where dealers with position sizes greater than dealer-position-limit will trade versus one another to restore their position to be equal to dealer-position-limit (restoring them within their limits).

### SIMULATING MARKET SCENARIOS

Market scenarios can be forced by clicking the below buttons:

Force Market Makers Short : this button instantaneously puts all market-makers beyond their short position limit.

Market Crash : this button instantaneously halves the value-investors' expectations in value and instagates a market crash.

Remove Value Investors : this button removes all value-investors from the system, leaving only the market-makers and the AI smartinvestors.


### Other controls

smart-investor-model : Select pretrained to use a pretrained model for the AI bots. In this case, the bots will not continue learning as they invest. Select not-pretrained to start each AI bot with randomly initialised neural networks. The bots will learn over time, choosing actions more deterministically as epsilon decreases.
 
## THINGS TO TRY

The focal point of this model is the price distribution. Note the Kurtosis and Skew of the distribution, and how it deviates from a normal distribution. Find which parameters create a distribution with a kurtosis above 3 (ie: a fat-tailed distribution), and notice how the skew is influenced by the prevailing market-maker positioning (when market-makers become too long, the skew becomes negative; and vice-versa when they become too short).

Try the market scenario buttons, for example to force a market crash. See if the price volatility increases after the crash. Alter the setup parameters to see what factors cause faster crashes than others.

Try altering the proportions of market-makers, value investors and smart investors. Also alter the market-disparity slider. In all cases, notice how these factors directly influence the resulting price distribution.

## EXTENDING THE MODEL


Try altering the code to introducing smart-investors in two waves, so that the second cohort of smart-investors gets initialised and begins training immediately after the first cohort finishes their training phase. Will the different waves learn differing strategies?

## CREDITS AND REFERENCES

Wilkinson, J.T (2022) *additional credits here?*

## HOW TO CITE

TO DO
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

computer
false
0
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105
Rectangle -7500403 true true 111 8 186 68
Rectangle -13791810 true false 119 15 179 60
Rectangle -7500403 true true 45 105 60 105
Rectangle -7500403 true true 142 67 157 82

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.2.2
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
<experiments>
  <experiment name="smartsize-vs-rewards" repetitions="1" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="170000"/>
    <metric>smartinvestor-rewards</metric>
    <enumeratedValueSet variable="mm-natsize">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="smartsize">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ninvestors">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="investor-limit-toggle">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bid-offer">
      <value value="0.4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-investor-limit">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ndealers">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nsmartinvestors">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mm-limit">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="prob-of-link">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mm-limit-toggle">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="verbose">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="market-disparity">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="trade-size-cap">
      <value value="20"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="dealerPosition-vs-skew" repetitions="100" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="20000"/>
    <metric>mean [ long - short ] of dealers</metric>
    <metric>skew</metric>
    <enumeratedValueSet variable="ndealers">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="trade-size-cap">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="prob-of-link">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nValueInvestors">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="smart-investor-model">
      <value value="&quot;Pretrained&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mm-limit-toggle">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="verbose">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nSmartInvestors">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="smartsize">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="market-disparity">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mm-PositionLimit">
      <value value="21"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bid-offer">
      <value value="1"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="Kurtosis-vs-smartVolume" repetitions="10" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="20000"/>
    <metric>kurtosis</metric>
    <metric>smartvolume</metric>
    <metric>dealervolume</metric>
    <metric>valuevolume</metric>
    <enumeratedValueSet variable="ndealers">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bid-offer">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mm-limit-toggle">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nValueInvestors">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="smart-investor-model">
      <value value="&quot;Pretrained&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="prob-of-link">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="verbose">
      <value value="false"/>
    </enumeratedValueSet>
    <steppedValueSet variable="nSmartInvestors" first="0" step="2" last="20"/>
    <enumeratedValueSet variable="smartsize">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="market-disparity">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="trade-size-cap">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mm-PositionLimit">
      <value value="21"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="getvolatility" repetitions="1" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="100000"/>
    <metric>rollingVol ( 1000 )</metric>
    <enumeratedValueSet variable="trade-size-cap">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nMarketMakers">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="prob-of-link">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nValueInvestors">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="smart-investor-model">
      <value value="&quot;Not Pretrained&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mm-limit-toggle">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="verbose">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nSmartInvestors">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="smartsize">
      <value value="2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="market-disparity">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="smart-holdingtime">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bid-offer">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mm-PositionLimit">
      <value value="5"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="experiment" repetitions="1" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="101000"/>
    <steppedValueSet variable="nValueInvestors" first="5" step="15" last="50"/>
    <enumeratedValueSet variable="smartsize">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="price-sensitivity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="smart-holdingtime">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bid-offer">
      <value value="1.1"/>
    </enumeratedValueSet>
    <steppedValueSet variable="nMarketMakers" first="6" step="2" last="10"/>
    <enumeratedValueSet variable="mm-limit-toggle">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="smart-investor-model">
      <value value="&quot;Not Pretrained&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="prob-of-link">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="verbose">
      <value value="false"/>
    </enumeratedValueSet>
    <steppedValueSet variable="nSmartInvestors" first="5" step="15" last="50"/>
    <enumeratedValueSet variable="market-disparity">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="trade-size-cap">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mm-PositionLimit">
      <value value="7"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="bid offer" repetitions="5" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="101000"/>
    <enumeratedValueSet variable="nValueInvestors">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="smartsize">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="price-sensitivity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="smart-holdingtime">
      <value value="250"/>
    </enumeratedValueSet>
    <steppedValueSet variable="bid-offer" first="0.5" step="0.1" last="2"/>
    <enumeratedValueSet variable="nMarketMakers">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mm-limit-toggle">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="smart-investor-model">
      <value value="&quot;Not Pretrained&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="prob-of-link">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="verbose">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nSmartInvestors">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="market-disparity">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="trade-size-cap">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mm-PositionLimit">
      <value value="7"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="fragmentation" repetitions="3" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="50100"/>
    <metric>count turtles</metric>
    <enumeratedValueSet variable="nValueInvestors">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="smartsize">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="price-sensitivity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="smart-holdingtime">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bid-offer">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nMarketMakers">
      <value value="10"/>
    </enumeratedValueSet>
    <steppedValueSet variable="prob-of-link" first="5" step="5" last="100"/>
    <enumeratedValueSet variable="smart-investor-model">
      <value value="&quot;Not Pretrained&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mm-limit-toggle">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="verbose">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="nSmartInvestors">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="market-disparity">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mm-PositionLimit">
      <value value="7"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="trade-size-cap">
      <value value="3"/>
    </enumeratedValueSet>
  </experiment>
</experiments>
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
