extensions [ py ]

globals [ a returns p0 p1 price-history histogram-num-bars obs-length totalsmartbuy smartvolume dealervolume valuevolume bidofferpaid bestbids bestoffers start-price meandeviation ]

breed [ valueinvestors valueinvestor ]
breed [ smartinvestors smartinvestor ]
breed [ dealers dealer ]
breed [ priceindep-buyers priceindep-buyer ] ; retail buyer

turtles-own [ expectation long short turn-taken? ]
links-own [ weight ]
dealers-own [ bid offer last-trade ] ; bid and offer prices
valueinvestors-own [ mynumber avgbuynum avgsellnum uncertainty ]
smartinvestors-own [ last-state confident? trade-holding-times state-memory positions transaction-prices ctransaction-price recent-price-history actions_index ]
priceindep-buyers-own [ lefttobuy ]


to setup

  clear-all

  ; random-seed 4

  set obs-length 500 ; length of price history used to train the smart valueinvestors RL. Keep at 500.
  set a 0.01 ;0.2 * bid-offer / mm-PositionLimit ; constant determining how much bid-offer increases as size increases. Model parameter

  set bestbids []
  set bestoffers []

  create-ordered-dealers nMarketMakers [
    set shape "house"
    set size 3
    fd 5
    rt 180
    set color grey
    set expectation 100
    set bid expectation - bid-offer / 2
    set offer expectation + bid-offer / 2
    create-links-with other dealers [ set weight random 100 ]
    set long 0
    set short 0
    set turn-taken? true
    set last-trade 100 ; initialise at the starting price
  ]

  create-ordered-valueinvestors nValueInvestors [
    set shape "person"
    set size 2
    fd 13
    rt 180
    set color blue
    set uncertainty 10 + random 5
    ifelse random 100 < 50 [
      set expectation random-normal ( 100 - market-disparity ) ( uncertainty )
    ][
      set expectation random-normal ( 100 + market-disparity ) ( uncertainty )
    ]
    create-links-with dealers [ set weight random 100 ]
    set long 0
    set short 0
    set turn-taken? false
  ]

  setup-python

  let model_built false
  let counter 0
  create-ordered-smartinvestors nSmartInvestors [
    set shape "person"
    set size 2
    fd 16
    rt 180
    set color red
    create-links-with dealers [ set weight random 100 ]
    set long 0
    set short 0
    set state-memory []

    py:set "params" []
    py:set "id" who
    py:set "lr" 1e-5
    py:set "state_size" obs-length
    py:set "eps_decay" 0.9995
    py:set "gamma" 0.99
    py:run "agents[id] = q.SmartTrader(lr, state_size, eps_decay=eps_decay, batch_size=64, gamma=gamma)"
    set model_built true

    set trade-holding-times []
    set positions []
    set transaction-prices []
    set recent-price-history []
    set actions_index []
    set confident? false
    set turn-taken? false

    if AI-investor-model = "Pretrained" [
      py:set "counter" counter
      py:run "modelname = 'DeepQtrader{}.pt'.format(counter)"
      py:run "agents[id].load_model(modelname)"
      set counter counter + 1
    ]
  ]

  ask links [ ; three boundary criteria: a turtle cannot have less than one connection to another dealer
    if weight > prob-of-link and count ( [ link-neighbors with [ breed = dealers ] ] of end2  ) > 1 and count ( [ link-neighbors with [ breed = dealers ] ] of end1 ) > 1 [
      die
    ]
    set color grey - 3
  ]

  set p0 price-level ; cache the price level
  set returns [] ; list for recording the returns
  set price-history []
  set meandeviation []

  reset-ticks

end


to setup-python
  py:setup "/Users/jameswilkinson/.conda/envs/NetLogoQLearn/bin/python"
  py:run "import deepQlearnV2 as q"
  py:run "agents = {}"
end


to go

  if not any? turtles with [ turn-taken? = false ]  [ ; if we're at the end of a round (make sure every valueinvestorhas had a turn at transacting)
    ask turtles [
      if breed = valueinvestors or breed = smartinvestors [
        set turn-taken? false
      ]
    ]
  ]

  ask one-of turtles with [ ( breed != dealers ) and ( turn-taken? = false ) ][
    set turn-taken? true
    if breed = valueinvestors [
      investor-act
    ]

    if breed = smartinvestors [
      smartinvestor-act
    ]
  ]

  ask dealers [ refresh-bidoffer ]

  let bestbid max [ bid ] of dealers
  let bestoffer min [ offer ] of dealers
  set bestbids lput bestbid bestbids
  set bestoffers lput bestoffer bestoffers

  if any? dealers with [ abs ( long - short ) > mm-PositionLimit ] [ ;
    ask one-of dealers with [ abs ( long - short ) > mm-PositionLimit ] [
      dealer-act
      ask dealers [ refresh-bidoffer ]
    ]
  ]

  if ( ticks mod 50 ) = 0 and not any? smartinvestors with [ not confident? ] [;and not any? smartinvestors with [ not confident? ] [ ; record the price changes every 50 ticks
    set p1 price-level
    set returns lput (p1 - p0) returns
    set p0 p1
    if start-price = 0 [
      set start-price mean [ expectation ] of dealers
    ]
  ]

  ; update price history
  set price-history lput mean [ expectation ] of dealers price-history
  ;set meandeviation lput ( mean [ expectation ] of dealers - mean [ expectation ] of valueinvestors ) meandeviation
  ask smartinvestors [
    set recent-price-history lput ( mean [ expectation ] of link-neighbors with [ breed = dealers ] - 100 ) recent-price-history ; -100 to make neural network inputs closer to 0
    if length recent-price-history > obs-length [
      set recent-price-history remove-item 0 recent-price-history ; keep the recent price history a fixed length
    ]
  ]

  ask links [
    set thickness thickness / 1.02
    set color color - 0.03
    if thickness < 0.2 [
      set color grey - 3
      set thickness 0
      set hidden? false
    ]
  ]

  ask dealers [
    if long - short > mm-PositionLimit [
      set color green
    ]
    if short - long > mm-PositionLimit [
      set color red
    ]
    if abs ( long - short ) < mm-PositionLimit [
      set color grey + 3
    ]
  ]

  tick
end


to smartinvestor-act

  py:set "id" who

  if ( py:runresult "agents[id].confident" ) [
    set confident? true
  ]

  ifelse length trade-holding-times > 0 and min trade-holding-times <= ticks [ ; if a trade has ended, close out deterministically and remember + learn whether the action from the state went well
    let index position ( min trade-holding-times ) trade-holding-times
    let action_index item index actions_index
    let ptransaction-price ( item index transaction-prices )
    let _position ( item index positions )
    if _position > 0 [
      smartinvestor-sell _position 0
    ]
    if _position < 0 [
      smartinvestor-buy -1 * _position 0
    ]

    set bidofferpaid bidofferpaid + bid-offer

    py:set "state" ( item index state-memory )
    py:set "next_state" recent-price-history
    py:set "reward" _position * ( ctransaction-price - ptransaction-price )  ; exclude the commission from the reward here if wanted... simple addition should do.
    py:set "action" action_index
    py:run "agents[id].remember(state, action, next_state, reward)"
    py:run "agents[id].learn()"

    ; finally, remove memory of the trade which is now closed
    set positions remove-item index positions
    set trade-holding-times remove-item index trade-holding-times
    set transaction-prices remove-item index transaction-prices
    set state-memory remove-item index state-memory

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

    ifelse buy_sell != "do nothing" [
      if buy_sell = "sell" [
        smartinvestor-sell trade_size trade_time
      ]
      if buy_sell = "buy" [
        smartinvestor-buy trade_size trade_time
      ]
    ][
      set transaction-prices lput 0 transaction-prices ; this is a "do nothing" transaction, reward will be zero so transaction_price doesn't matter
      set positions lput 0 positions
    ]
  ]
end


to refresh-bidoffer ; dealer procedure


  let axe-adj min list bid-offer a * sensitivity-function ( short - long  )

  set expectation last-trade + axe-adj

  set offer ( expectation + bid-offer / 2 )
  set bid ( expectation - bid-offer / 2 )

end


to dealer-act
  if ( mm-limit-toggle and ( long - short > mm-PositionLimit ) ) [
    dealersell
  ]
  if ( mm-limit-toggle and ( short - long > mm-PositionLimit ) ) [
    dealerbuy
  ]
end


to investor-act ;

  ;; four types of action:
  ;;  1) Get longer
  ;;  2) Get shorter

  let bestbid [ bid ] of max-one-of ( link-neighbors with [ breed = dealers ] ) [ bid ] ; best bid in the market
  let bestoffer [ offer ] of min-one-of ( link-neighbors with [ breed = dealers ] ) [ offer ] ; best offer in the market

  let gain-from-buy ( expectation - bestoffer )
  let gain-from-sell ( bestbid - expectation )

  ;; 1) Get shorter
  if ( bestbid > expectation ) and ( gain-from-buy < gain-from-sell ) [ ; if the price is real, and higher than expectation, sell
    valueInvestorSell
  ]

   ;; 2) Get longer
  if ( bestoffer < expectation ) and ( gain-from-sell < gain-from-buy ) [ ; if the price is real, and lower than expectation, buy
    valueInvestorBuy
  ]

end


to smartinvestor-sell [ tradesize trade_time ]

  let bestdealer max-one-of link-neighbors with [ breed = dealers ] [ short - long ] ; naturally, this will select the dealer with most room
  let bestbid [ bid ] of bestdealer
  let max-natural-size  max list ( 0 ) ( [ short - long + mm-PositionLimit ] of bestdealer ) ; amount of room the dealer can buy without breeching limits

  let excess-bid bestbid - a * ( tradesize - max-natural-size ) ; CHANGE
  let excess-size max list (0) ( tradesize - max-natural-size )
  let size-adj-bid ( ( bestbid * min list tradesize max-natural-size ) + ( excess-bid * excess-size ) ) / tradesize ; weighted mean of prices

  set short short + tradesize
  ask dealers with [ in-link-neighbor? bestdealer ] [
    set last-trade size-adj-bid
  ]
  ask bestdealer [
    set last-trade size-adj-bid
  ]

  ifelse length trade-holding-times > 0 and min trade-holding-times <= ticks [ ; if it was an old trade being closed, remove the trade stats from the traders memory
    set ctransaction-price size-adj-bid
  ][ ; else create a new part in the memory
    set positions lput ( -1 * tradesize ) positions ; sell is negative position
    set transaction-prices lput size-adj-bid transaction-prices
  ]

  ; printing stats
  if verbose [ print ( word breed  who "sells " tradesize " to dealer" [ who ] of bestdealer " @" size-adj-bid ) ]

  ask bestdealer [ ; update the counterparty dealer's stats and refresh their bid offer
    set long long + tradesize
  ]

  ask my-links with [ end1 = bestdealer ] [
    set thickness min list 2 tradesize * 0.2
    set hidden? false
    set color red
  ]

  set smartvolume smartvolume + tradesize

end


to smartinvestor-buy [ tradesize trade_time ]

  let bestdealer max-one-of link-neighbors with [ breed = dealers ] [ long - short ] ; naturally, this will select the dealer with most room
  let bestoffer [ offer ] of bestdealer
  let max-natural-size  max list ( 0 ) ( [ long - short + mm-PositionLimit ] of bestdealer ) ; amount of room the dealer can buy without breeching limits

  let excess-offer bestoffer + a * ( tradesize - max-natural-size )
  let excess-size max list (0) ( tradesize - max-natural-size )
  let size-adj-offer ( ( bestoffer * min list tradesize max-natural-size ) + ( excess-offer * excess-size ) ) / tradesize ; weighted mean of prices

  set long long + tradesize
  ask dealers with [ in-link-neighbor? bestdealer ] [
    set last-trade size-adj-offer
  ]
  ask bestdealer [
    set last-trade size-adj-offer
  ]

  ifelse length trade-holding-times > 0 and min trade-holding-times <= ticks [ ; if it was an old trade being closed, remove the trade stats from the traders memory
    let index position ( min trade-holding-times ) trade-holding-times
    set ctransaction-price size-adj-offer
  ][ ; else create a new part in the memory
    set positions lput tradesize positions
    set transaction-prices lput size-adj-offer transaction-prices
  ]

  ; printing stats
  if verbose [ print ( word breed  who "buys " tradesize " from dealer" [ who ] of bestdealer " @" size-adj-offer ) ]

  ask bestdealer [ ; update the counterparty dealer's stats and refresh their bid offer
    set short short + tradesize
  ]

  ask my-links with [ end1 = bestdealer ] [
    set thickness min list 2 tradesize * 0.2
    set hidden? false
    set color green
  ]

  set smartvolume smartvolume + tradesize

end


to dealerbuy ; valueinvestorprocedure
  let bestdealer max-one-of link-neighbors with [ breed = dealers ] [ long - short ] ; naturally, this will select the dealer with most room
  let bestoffer [ offer ] of bestdealer
  let max-natural-size  max list ( 0 ) ( [ long - short + mm-PositionLimit ] of bestdealer ) ; amount of room the dealer can buy without breeching limits
  let tradesize min list ( trade-size-cap ) ( short - long - mm-PositionLimit )
  let excess-offer bestoffer + a * ( tradesize - max-natural-size)
  let excess-size max list (0) ( tradesize - max-natural-size )
  let size-adj-offer ( ( bestoffer * min list tradesize max-natural-size ) + ( excess-offer * excess-size ) ) / tradesize ; weighted mean of prices

  set long long + tradesize
  ask dealers with [ in-link-neighbor? bestdealer ] [
    set last-trade size-adj-offer
  ]
  ask bestdealer [
    set last-trade size-adj-offer
  ]

  ; printing stats
  if verbose [ print ( word breed who "buys " tradesize " from dealer" [ who ] of bestdealer " @" size-adj-offer ) ]

  ask bestdealer [ ; update the counterparty dealer's stats and refresh their bid offer
    set short short + tradesize
  ]

  ask my-links with [ end1 = bestdealer ] [
    set thickness min list 2 tradesize * 0.2
    set hidden? false
    set color green
  ]

  set dealervolume dealervolume + tradesize
end


to dealersell
  let bestdealer max-one-of link-neighbors with [ breed = dealers ] [ short - long ] ; naturally, this will select the dealer with most room
  let bestbid [ bid ] of bestdealer
  let max-natural-size  max list ( 0 ) ( [ short - long + mm-PositionLimit ] of bestdealer ) ; amount of room the dealer can buy without breeching limits
  let tradesize min list ( trade-size-cap ) ( long - short - mm-PositionLimit )
  let excess-bid bestbid - a * ( tradesize - max-natural-size)
  let excess-size max list (0) ( tradesize - max-natural-size )
  let size-adj-bid ( ( bestbid * min list tradesize max-natural-size ) + ( excess-bid * excess-size ) ) / tradesize ; weighted mean of prices

  set short short + tradesize
  ask dealers with [ in-link-neighbor? bestdealer ] [
    set last-trade size-adj-bid
  ]
  ask bestdealer [
    set last-trade size-adj-bid
  ]

  ; printing stats
  if verbose [ print ( word breed who "sells " tradesize " to dealer" [ who ] of bestdealer " @" size-adj-bid ) ]

  ask bestdealer [ ; update the counterparty dealer's stats and refresh their bid offer
    set long long + tradesize
  ]

  ask my-links with [ end1 = bestdealer ] [
    set thickness min list 2 tradesize * 0.2
    set hidden? false
    set color red
  ]

  set dealervolume dealervolume + tradesize
end


to valueInvestorBuy ; valueinvestorprocedure

  let bestdealer max-one-of link-neighbors with [ breed = dealers ] [ long - short ] ; naturally, this will select the dealer with most room
  let bestoffer [ offer ] of bestdealer
  let max-natural-size  max list ( 0 ) ( [ long - short + mm-PositionLimit ] of bestdealer ) ; amount of room the dealer can buy without breeching limits
  let tradesize max list ( 0 ) ( short - long ) ; if positioned wrong way, we want to close out the position AND put on a trade

  set tradesize min list ( trade-size-cap ) ( tradesize + ( expectation - bestoffer ) / ( uncertainty + a / 2 ) )

  let excess-offer bestoffer + a * ( tradesize - max-natural-size )
  let excess-size max list (0) ( tradesize - max-natural-size )
  let size-adj-offer ( ( bestoffer * min list tradesize max-natural-size ) + ( excess-offer * excess-size ) ) / tradesize ; weighted mean of prices

  set long long + tradesize
  ask dealers with [ in-link-neighbor? bestdealer ] [
    set last-trade size-adj-offer
  ]
  ask bestdealer [
    set last-trade size-adj-offer
  ]

  ; printing stats
  if verbose [ print ( word breed  who "buys " tradesize " from dealer" [ who ] of bestdealer " @" size-adj-offer " target" expectation) ]

  ask bestdealer [ ; update the counterparty dealer's stats and refresh their bid offer
    set short short + tradesize
  ]

  ask my-links with [ end1 = bestdealer ] [
    set thickness min list 2 tradesize * 0.2
    set hidden? false
    set color green
  ]

  set valuevolume valuevolume + tradesize
end


to valueInvestorSell ; valueinvestorprocedure

  let bestdealer max-one-of link-neighbors with [ breed = dealers ] [ short - long ] ; naturally, this will select the dealer with most room
  let bestbid [ bid ] of bestdealer
  let max-natural-size  max list ( 0 ) ( [ short - long + mm-PositionLimit ] of bestdealer ) ; amount of room the dealer can buy without breeching limits

  let tradesize max list ( 0 ) ( long - short ) ; if positioned wrong way, we want to close out the position AND put on a trade

  ; if valueinvestoris not selling because of inventory limits, then act as normal. Else, trade to get within limits again
  set tradesize min list ( trade-size-cap ) ( tradesize + ( bestbid - expectation ) / ( uncertainty + a / 2 ) )

  let excess-bid bestbid - a * ( tradesize - max-natural-size) ; price to take the extra at
  let excess-size max list (0) ( tradesize - max-natural-size )
  let size-adj-bid ( ( bestbid * min list tradesize max-natural-size ) + ( excess-bid * excess-size ) ) / tradesize ; weighted mean of prices

  set short short + tradesize
  ask dealers with [ in-link-neighbor? bestdealer ] [
    set last-trade size-adj-bid
  ]
  ask bestdealer [
    set last-trade size-adj-bid
  ]

  ; printing stats
  if verbose [ print ( word breed  who "sells " tradesize " to dealer" [ who ] of bestdealer " @" size-adj-bid " target" expectation) ]

  ask bestdealer [ ; update the counterparty dealer's stats and refresh their bid offer
    set long long + tradesize
  ]

  ask my-links with [ end1 = bestdealer ] [
    set thickness min list 2 tradesize * 0.2
    set hidden? false
    set color red
  ]

  set valuevolume valuevolume + tradesize

end


to plot-normal [ histogram-bins ]
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


to-report smile [ x ] ; report standard dev of normal distribution fitting returns geq than x
  let filt_returns []
  ifelse x <= 0 [
    set filt_returns filter [ i -> i <= x ] returns
  ][
    set filt_returns filter [ i -> i >= x ] returns
  ]

  ;let filt_returns filter [ i -> ( abs i ) >= ( abs x ) ] returns
  let filt_returns_doublesided ( sentence ( filt_returns ) ( map [ i -> i * -1 ] filt_returns ) )
  ifelse length filt_returns > 2 [
    let var variance filt_returns_doublesided
    report sqrt var
  ][
    report "None"
  ]
end

to plot-smile
  clear-plot

  if length returns > 50 [
    let xrange max list int ( min returns - 1 )  int ( max returns + 1 )
    set-plot-x-range -1 * xrange xrange
    let totalvar variance returns
    let stepper plot-x-min
    while [ stepper < plot-x-max ][
      if smile stepper != "None" [
        plotxy stepper ( smile stepper )
      ]
      set stepper stepper + 0.1
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


to move-market ; valueinvestorprocedure
  ;setup-investors
  ask valueinvestors [
    set expectation expectation * 0.8 ; force a market crash
  ]
end


to force-dealers-short ; when called, forces dealers to get limit-long
  ask dealers [
    set short long + mm-PositionLimit * 1.5
  ]
end


to introduce-price-indep-buyer ; introduce a new valueinvestortemporarily to the system, with a certain amount to buy regardless of the price
  create-ordered-priceindep-buyers 1
  ask priceindep-buyers [ ; This probably needs making more robust.
    set color red
    set lefttobuy 100
    set size lefttobuy / 100 ; make big and we'll decrease its size as it sells. This makes a nice intuitive visualisation
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


to-report benchmark-level
  report sum [ mynumber ] of valueinvestors
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



to save-model
  let counter 0
  ask smartinvestors [
    py:set "counter" counter
    py:run "modelname = 'DeepQtrader{}.pt'.format(counter)"
    py:set "id" who
    py:run "agents[id].save_model(modelname)"
    set counter counter + 1
  ]
end


to-report sensitivity-function [ x ]
  report x ; linear
end


to-report smartinvestor-rewards
  let meanreward ( py:runresult "sum([agents[id].totalreward for id in agents.keys()])" ) / nSmartInvestors
  report meanreward
end


to-report smartinvestor-epsilon
  let epsilon py:runresult "sum([agents[id].epsilon for id in agents.keys()])" / nSmartInvestors
  report epsilon
end


to-report z-score [x _mean var ]; z scores x on normal distro with _mean and var
  report abs( x - _mean ) / sqrt ( var )
end
@#$#@#$#@
GRAPHICS-WINDOW
922
10
1296
385
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
14
123
186
156
nValueInvestors
nValueInvestors
0
50
50.0
1
1
NIL
HORIZONTAL

BUTTON
13
24
79
57
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
585
13
644
58
OFFER
best-offer
2
1
11

MONITOR
515
13
580
58
BID
best-bid
2
1
11

BUTTON
94
73
157
106
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
15
72
80
105
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
407
65
907
287
Price
ticks
price
0.0
10.0
0.0
10.0
true
true
"set-plot-y-range 99 101" "set-plot-x-range ( max list ( 0 ) ( ticks - 2000 ) ) ticks + 10"
PENS
"market bid" 1.0 0 -13345367 true "" "let bestbid [ bid ] of max-one-of dealers [ bid ]\nlet bestoffer [ offer ] of min-one-of dealers [ offer ]\nifelse bestbid = -1e11 [ \nset-plot-pen-color white\nplot bestoffer - bid-offer\n][\nset-plot-pen-color blue\nplot bestbid\n]"
"market offer" 1.0 0 -2674135 true "" "let bestoffer [ offer ] of min-one-of dealers [ offer ]\nlet bestbid [ bid ] of max-one-of dealers [ bid ]\nifelse bestoffer > 1e10 [\nset-plot-pen-color white \nplot bestbid + bid-offer\n][\nset-plot-pen-color red\nplot bestoffer\n]"

SLIDER
13
257
185
290
bid-offer
bid-offer
0
2
0.5
0.1
1
NIL
HORIZONTAL

SLIDER
13
305
185
338
mm-PositionLimit
mm-PositionLimit
0
100
7.0
1
1
NIL
HORIZONTAL

PLOT
407
292
909
459
Realised Return
50 tick change
Count
-10.0
10.0
0.0
10.0
true
true
"clear-plot\nset-plot-x-range ( -1 * bid-offer * 20 ) ( bid-offer * 20 )\nset histogram-num-bars 2 * ( plot-x-max - plot-x-min ) / bid-offer" ""
PENS
"Observed" 1.0 1 -16777216 true "set-histogram-num-bars histogram-num-bars" "set-histogram-num-bars histogram-num-bars\nhistogram returns"
"Normal" 1.0 2 -5298144 true "" "plot-normal histogram-num-bars"

SLIDER
16
166
188
199
nMarketMakers
nMarketMakers
2
20
10.0
1
1
NIL
HORIZONTAL

BUTTON
210
340
324
373
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

SWITCH
205
83
308
116
verbose
verbose
1
1
-1000

MONITOR
699
13
780
58
mean view
mean [ expectation ] of valueinvestors
1
1
11

TEXTBOX
211
258
361
276
Market Scenarios\n
11
0.0
1

BUTTON
210
289
377
322
Force Market Makers short
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
13
395
185
428
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
16
212
188
245
nSmartInvestors
nSmartInvestors
0
50
50.0
1
1
NIL
HORIZONTAL

SLIDER
13
348
185
381
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
13
441
185
474
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
210
390
366
423
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

PLOT
408
467
694
587
smartinvestor profit
ticks
total PnL
0.0
10.0
0.0
10.0
true
false
"" "; set-plot-x-range ( max list ( 0 ) ( ticks - 2000 ) ) ticks + 10"
PENS
"default" 1.0 0 -16777216 true "" "if not any? smartinvestors with [ not confident? ] [\nplot smartinvestor-rewards\n]"

PLOT
708
468
908
588
average epsilon
NIL
NIL
0.0
10.0
0.0
1.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot smartinvestor-epsilon"

CHOOSER
205
189
368
234
AI-investor-model
AI-investor-model
"Pretrained" "Not Pretrained"
0

MONITOR
762
322
827
367
Kurtosis
kurtosis
3
1
11

MONITOR
439
322
506
367
Skew
skew
3
1
11

PLOT
921
389
1294
525
Volatility
NIL
NIL
0.0
10.0
0.0
0.1
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "if length price-history > 500 * 30 [\n  plot rollingVol 500 30\n]"

BUTTON
101
27
172
60
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
204
135
352
168
mm-limit-toggle
mm-limit-toggle
1
1
-1000

MONITOR
794
15
860
60
mean px
mean price-history
1
1
11

@#$#@#$#@
## WHAT IS IT?

This model simulates the evolution of the price of a financial instrument by modelling a market with an agent based approach. The model comprises of a set of investors interracting versus market-makers.

The goal of this model is to simulate a free market with static information, which allows investigation into the pricing dynamics of a market, the impact of market structure on market function, the diffusion of information via price action within a market, and more.

The model comprises of market-makers (house-shaped turtles) which provide prices, value-investors (blue person-shaped turtles) who buy or sell based on pre-determined views on the market (which can be interpreted as analyst expectations), and AI smart-traders (red person-shaped turtles) which are trained using reinforcement learning methods. 

Trades occur between investors and market-makers down the visible link turtles, which flash green when an investor buys, and red when an investor sells.


## HOW IT WORKS

Market-makers begin with their prices centered at 100.

Value-investors are initialised with an "opinion" about the market price, and an uncertainty about their opinion. If their opinion is higher than the prevailing market price, they will buy from a market maker (and vice versa if their opinion is lower than the prevailing market price). Once a trade occurs, the price of the trade is "published", and all market-makers adjust their prices towards the published trade price. When an investor transacts, they will transact versus the market-maker with the most attractive price.

Value-investors will increase the size of their trade proportionally to the difference between their opinion and the transactable market price, and inversly proportionally to their uncertainty up to a maximum trade size (trade-size-cap). Market-makers have a soft limit (marketMaker-limit) to how long or short they can be at any moment. A trade that puts the market-maker outside of the marketMaker-limit will result in the trade commission increasing linearly with the resulting limit-breech.

Smart-investors are AI bots trained to profit off of fixed-sized trades. Trade sizes are fixed to be smartsize.

Market-makers can also act as investors and trade versus one another, a behaviour that is triggered when the mm-limit-toggle is turned on and when market-makers breech their soft limit. When this criteria is met, market-makers sell or buy versus the other market-makers in a size that restors them inside their soft limits.



## HOW TO USE IT

You can intuitively adjust the structure of the market, varying the number of market makers, the number of value investors and the number of AI smartinvestors.

### Slider functions

nValueInvestors : the number of value-investors (blue person turtles) initialised in the model

nMarketMakers : the number of market-makers (house turtles) initialised in the model

nSmartInvestors : the number of AI investors (red person turtles) initialised in the model

bid-offer : the commission associated with trading. This is the difference between a market-maker's offer price and bid price.

price-sensitivity: When a market maker observes a trade that has happened away from them, this parameter represent the percentage proprtion of how fast they move their mid price to the trade price. If 100, market makers will move their mid-prices directly to any new trade price. If 50, the market makers will move their prices half way between their original price, and the trade price.

mm-PositionLimit : the position size a market maker can accumulate before they begin skewwing their price, or (if the mm-limit-toggle is turned on) before a market-maker will transact with other market-makers to reduce their position size.

prob-of-link : the probability of a link forming between a turtle and a market maker. NB: This is constrained so every turtle is connected to at least one market maker.

trade-size-cap : the limit to the size of a single trade allowed to be executed by a value investor, and (if the mm-limit-toggle is turned on) between market makers.

market-disparity : the difference between the two-normal distribution means from which value-investor opinions are drawn from. When the market-disparity is increased, the two normal distributions become centered equidistant on either side of 100 by this amount. When market-disparity is zero, all value investors will have their expectation drawn from a normal distribution centered at 100. If the market disparity is 20, approximately half of the value-investors have their expectations drawn from a normal distribution centered at 120, whilst the other half have their normal distribution centered at 80.

smartsize : the fixed trade size allowed by a smartinvestor (the AI bot)

### Toggles

verbose : turn ON to see a print out of the trades happening in the market

mm-limit-toggle : turn ON to trigger market-maker behaviour, where market-makers with position sizes greater than mm-PositionLimit will trade versus one another to restore their position to be equal to mm-PositionLimit (restoring them within their limits).

### Buttons

Market scenarios can be forced by clicking the below buttons:

Force Market Makers Short : this button instantaneously puts all market-makers beyond their short position limit.

Market Crash : this button instantaneously halves the value-investors' expectations in value and instagates a market crash.

Remove Value Investors : this button removes all value-investors from the system, leaving only the market-makers and the AI smartinvestors.


### Other controls

smart-investor-model : Select pretrained to use a pretrained model for the AI bots. In this case, the bots will not continue learning as they invest. Select not-pretrained to start each AI bot with randomly initialised neural networks. The bots will learn over time, choosing actions more deterministically as epsilon decreases.
 
## THINGS TO NOTICE

The focal point of this model is the price distribution. Note the Kurtosis and Skew of the distribution, and how it deviates from a normal distribution. Find which parameters create a Kurtosis of above 3 (ie: a fat-tailed distribution), and notice how the skew is influenced by the prevailing market-maker positioning (when market-makers become too long, the skew becomes negative; and vice-versa when they become too short).
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
