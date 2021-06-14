//+------------------------------------------------------------------+
//|                                               PredHLStrategy.mq4 |
//|                Copyright 2021, J. Lee, J. Huang, O. Lin, V. Guo. |
//|                                                  http://qffc.org |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, J. Lee, J. Huang, O. Lin, V. Guo."
#property link      "http://qffc.org"
#property version   "1.00"
#property strict

// Define the product symbol variables.
string            TPSymbol = "";          // The current product symbol.

// Set the trade control variables.
int               pHH = 0;                // The previous hour.
int               cHH = 0;                // The current hour.
int               tradeDay = 100;         // The total trading days.
double            TPLot = 1.0;            // The total transaction volume.
string            TPMagic = 88888;        // The magic value of the product.
double            cPrice = 0;             // The current price of the product.
double            sPrice = 0;             // The string of the current price of the product.
double            pClose = 0;             // The previous close for the current product.
double            predO = 0;              // The predicted open for the next day.
double            predC = 0;              // The predicted close for the next day.
double            predH = 0;              // The predicted high for the next day.
double            predL = 0;              // The predicted low for the next day.
double            nBuy_Pass;              // The threshold of buying.
double            nSell_Pass;             // The threshold of selling.
int               sl = 200;               // The stop loss.
int               tp = 250;               // The target profits.
bool              bBuy_Stop = false;      // The signal for stopping buy order.
bool              bBuy_Pass = false;      // The signal for buy order.
string            sBuy_Signal = "NOSIG";  // The signal value for buy.
bool              bSell_Stop = false;     // The signal for stopping sell order.
bool              bSell_Pass = false;     // The signal for sell order.
string            sSell_Signal = "NOSIG"; // The signal value for sell.

// Set the predicted values directory.
string            DataDir = "FXStrategy"; // The data directory.
string            HLCOPredFileName = "";  // The predicted HLCO filename.
int               HLCOPredFileHandle;     // The predicted HLCO file handle.

// Initialize.
int OnInit()
{
    // Get the symbol of the current product.
    TPSymbol = Symbol();
    // Initialize the trade hour.
    pHH = 0;
    cHH = 0;
    bBuy_Stop = false;
    bBuy_Pass = false;
    sBuy_Signal = "NOSIG";
    bSell_Stop = false;
    bSell_Pass = false;
    sSell_Signal = "NOSIG";

    // Get the previous close.
    pClose = iClose(TPSymbol, PERIOD_D1, 1);
    // Get the prediction.
    HLCOPredFileName = TPSymbol + "_Pred.csv";
    HLCOPredFileHandle = FileOpen(DataDir + "//Preds//" + HLCOPredFileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
    // Get the value.
    predO = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
    predH = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
    predL = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
    predC = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
    // Close the file.
    FileClose(HLCOPredFileHandle);
    // Rectify the predicted value.
    predH = RectifyPred(pClose, predO, predH);
    predL = RectifyPred(pClose, predO, predL);
    // Get the buy and sell pass.
    nSell_Pass = predH;
    nBuy_Pass = predL;

    // if (TimeHour(TimeLocal()) < 21)
    // {
    //     // Get the prediction.
    //     HLCOPredFileName = (tradeDay - 1) + "_" + TPSymbol + "_Pred.csv";
    //     HLCOPredFileHandle = FileOpen(DataDir + "//Preds//" + HLCOPredFileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
    //     // Get the value.
    //     predO = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
    //     predH = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
    //     predL = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
    //     predC = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
    //     // Close the file.
    //     FileClose(HLCOPredFileHandle);
    //     // Get the previous close.
    //     pClose = iClose(TPSymbol, PERIOD_D1, 1);
    //     // Rectify the predicted value.
    //     predH = RectifyPred(pClose, predO, predH);
    //     predL = RectifyPred(pClose, predO, predL);
    //     // Get the buy and sell pass.
    //     nSell_Pass = predH;
    //     nBuy_Pass = predL;
    //     // Decrease the trading day.
    //     tradeDay = tradeDay - 1;
    // }

    // Return the result for initialized successfully.
    return(INIT_SUCCEEDED);
}

// Deinit.
void OnDeinit(const int reason)
{
    // Deinitialize all the variables.
    pHH = 0;
    cHH = 0;
    bBuy_Stop = false;
    bBuy_Pass = false;
    sBuy_Signal = "NOSIG";
    bSell_Stop = false;
    bSell_Pass = false;
    sSell_Signal = "NOSIG";
    TPSymbol = "";
}

// Main Function.
void OnTick()
{
    // Get the current time hour.
    cHH = TimeHour(TimeLocal());
    // Refresh the buy and sell status every morning.
    if ((pHH == 6) && (cHH == 7))
    {
        bBuy_Stop = false;
        bBuy_Pass = false;
        sBuy_Signal = "NOSIG";
        bSell_Stop = false;
        bSell_Pass = false;
        sSell_Signal = "NOSIG";
    }
    // Check whether reach the sleep time.
    if (TimeHour(TimeLocal()) >= 21)
    {
        bBuy_Stop = true;
        sBuy_Signal = "SLEEPING";
        bSell_Stop = true;
        sSell_Signal = "SLEEPING";

        if (TimeHour(TimeLocal()) == 21)
        {
            // Get the prediction.
            HLCOPredFileName = TPSymbol + "_Pred.csv";
            HLCOPredFileHandle = FileOpen(DataDir + "//Preds//" + HLCOPredFileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
            // Get the value.
            predO = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
            predH = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
            predL = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
            predC = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
            // Close the file.
            FileClose(HLCOPredFileHandle);
            // Get the previous close.
            pClose = iClose(TPSymbol, PERIOD_D1, 0);
            // Rectify the predicted value.
            predH = RectifyPred(pClose, predO, predH);
            predL = RectifyPred(pClose, predO, predL);
            // Get the buy and sell pass.
            nSell_Pass = predH;
            nBuy_Pass = predL;

            // // Get the prediction.
            // HLCOPredFileName = (tradeDay - 1) + "_" + TPSymbol + "_Pred.csv";
            // HLCOPredFileHandle = FileOpen(DataDir + "//Preds//" + HLCOPredFileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
            // // Get the value.
            // predO = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
            // predH = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
            // predL = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
            // predC = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
            // // Close the file.
            // FileClose(HLCOPredFileHandle);
            // // Get the previous close.
            // pClose = iClose(TPSymbol, PERIOD_D1, 0);
            // // Rectify the predicted value.
            // predH = RectifyPred(pClose, predO, predH);
            // predL = RectifyPred(pClose, predO, predL);
            // // Get the buy and sell pass.
            // nSell_Pass = predH;
            // nBuy_Pass = predL;
            // // Decrease the trading day.
            // tradeDay = tradeDay - 1;
        }
    }
    // Prevent more buy order if there are some outstanding buy orders.
    if (nActiveBuyOrder() >= 1)
    {
        bBuy_Stop = true;
        sBuy_Signal = "STOPBUY";
    }
    // Prevent more sell order if there are some outstanding sell orders.
    if (nActiveSellOrder() >= 1)
    {
        bSell_Stop = true;
        sSell_Signal = "STOPSELL";
    }
    // Ask the seller's price.
    cPrice = Ask;
    sPrice = DoubleToString(cPrice, 5);
    // Check whether to make the buy order.
    if (!bBuy_Stop && !bBuy_Pass && (cPrice <= nBuy_Pass))
    {
        bBuy_Pass = true;
        sBuy_Signal = "BUYING";
    }
    // Check whether to send the buy order.
    if (!bBuy_Stop && bBuy_Pass)
    {
        // Send the buy order.
        BuyOrder(TPLot, sl, tp, TPSymbol + "BUY", TPMagic);
        // Change the buy status.
        sBuy_Signal = "BUYORDERED";
        bBuy_Stop = true;
    }
    // Print the current status.
    Print(TPSymbol, " : Price = ", sPrice, " QPL+ = ", nSell_Pass, " QPL- = ", nBuy_Pass, " BUYSIGNAL = ", sBuy_Signal);
    // Ask the buyer's price.
    cPrice = Bid;
    sPrice = DoubleToString(cPrice, 5);
    // Check whether to make the sell order.
    if (!bSell_Stop && !bSell_Pass && (cPrice >= nSell_Pass))
    {
        bSell_Pass = true;
        sSell_Signal = "SELLING";
    }
    // Check whether to send the sell order.
    if (!bSell_Stop && bSell_Pass)
    {
        // Check whether there exist one buy order.
        for (int i = 0; i < OrdersTotal(); i++)
        {
            // Check all the sell order.
            if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
            {
                if (OrderType() == OP_BUY)
                {
                    // Send the sell order.
                    SellOrder(TPLot, sl, tp, TPSymbol + "SELL", TPMagic);
                    // Change the sell status.
                    sSell_Signal = "SELLORDERED";
                    bSell_Stop = true;
                    // Quit sell.
                    break;
                }
            }
        }
        // Check whether complete the sell order.
        if (!bSell_Stop)
        {
            bSell_Pass = false;
            sSell_Signal = "NOSIG";
        }
    }
    // Print the current status.
    Print(TPSymbol, " : Price = ", sPrice, " QPL+ = ", nSell_Pass, " QPL- = ", nBuy_Pass, " SELLSIGNAL = ", sSell_Signal);
    // Update the previous hour.
    pHH = cHH;
    // Print the order status.
    Print("Buy Order: ", IntegerToString(nActiveBuyOrder(), 10));
    Print("Sell Order: ", IntegerToString(nActiveSellOrder(), 10));
    // Sleep for 15 seconds.
    Sleep(15000);
}

// Create the function to rectify the prediction.
double RectifyPred(double cClose, double pOpen, double x)
{
    // Get the loss between the previous close and predicted open.
    double loss = pOpen - cClose;
    // Get the rectified prediction value.
    return (x - loss);
}

// Create the function for active sell order checking.
int nActiveSellOrder()
{
    // Initialize the sell order counter.
    int sCounter = 0;
    // Check all the sell order.
    for (int i = 0; i < OrdersTotal(); i++)
    {
        // Check all the sell order.
        if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
            if (OrderType() == OP_SELL)
            {
                sCounter++;
            }
        }
    }
    // Return the number of outstanding sell order.
    return sCounter;
}

// Create the function for active buy order checking.
int nActiveBuyOrder()
{
    // Initialize the buy order counter.
    int bCounter = 0;
    // Check all the buy order.
    for (int i = 0; i < OrdersTotal(); i++)
    {
        // Check all the buy order.
        if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
            if (OrderType() == OP_BUY)
            {
                bCounter++;
            }
        }
    }
    // Return the number of the outstanding buy order.
    return bCounter;
}

// Create the function for send the sell order.
int SellOrder(double Lots, double sloss, double tprice, string comment, int magic2)
{
    // Initialize the ticket for sell order.
    int nticket = 0;
    // Send the sell order.
    nticket = OrderSend(TPSymbol, OP_SELL, Lots, Bid, 0, Ask + sloss * Point, Bid - tprice * Point, comment, magic2, 0, Green);
    // Return the ticket.
    return nticket;
}

// Create the function for send the buy order.
int BuyOrder(double Lots, double sloss, double tprice, string comment, int magic2)
{
    // Initialize the ticket for buy order.
    int nticket = 0;
    // Send the buy order.
    nticket = OrderSend(TPSymbol, OP_BUY, Lots, Ask, 0, Bid - sloss * Point, Ask + tprice * Point, comment, magic2, 0, Red);
    // Return the ticket.
    return nticket;
}