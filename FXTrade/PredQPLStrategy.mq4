//+------------------------------------------------------------------+
//|                                              PredQPLStrategy.mq4 |
//|                Copyright 2021, J. Lee, J. Huang, O. Lin, V. Guo. |
//|                                                  http://qffc.org |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, J. Lee, J. Huang, O. Lin, V. Guo."
#property link      "http://qffc.org"
#property version   "1.00"
#property strict

// Define the global variables for QPLs.
int                 maxEL = 21;             // The maximum energy level.
int                 maxTS = 2048;           // The maximum time series for QPL.
double              p3 = 1.0 / 3.0;         // The power of the K value.
// Declare the variables to compute the NQPR.
int                 eL, d, TSsize, nQ, maxQno, nR, maxRno, tQno;
double              auxR, maxQ, r0, r1, rn1;
double              dr, Lup, Ldw, L, mu, sigma;
bool                bFound;
// Variables used in Cardano's Method.
double              p, q, u, v;
// Declare time series array.
static double       DT_OP[2048];
static double       DT_HI[2048];
static double       DT_LO[2048];
static double       DT_CL[2048];
static double       DT_RT[2048];
// Declare array for Quantum Price Wavefunction.
double              Q[100];
double              NQ[100];
double              r[100];
// Declare array for NQPR related arrays.
double              QFEL[21];               // The QFEL for the current product.
double              QPR[21];                // The QPR for the current product.
static double       NQPR[21];               // The NQPR for the current product.
double              PQPL[21];               // The positive QPL for the current product.
double              NQPL[21];               // The negative QPL for the current product.
double              K[21];                  // The K value for the current product.

// Define the product symbol variables.
string              TPSymbol = "";          // The current product symbol.

// Set the trade control variables.
int                 pHH = 0;                // The previous hour.
int                 cHH = 0;                // The current hour.
int                 tradeDay = 100;         // The total trading days.
double              TPLot = 1.0;            // The total transaction volume.
string              TPMagic = 88888;        // The magic value of the product.
double              cPrice = 0;             // The current price of the product.
double              sPrice = 0;             // The string of the current price of the product.
double              predO = 0;              // The predicted open for the next day.
double              pClose = 0;             // The previous close for the current product.
double              nBuy_Pass;              // The threshold of buying.
double              nSell_Pass;             // The threshold of selling.
int                 sl = 200;               // The stop loss.
int                 tp = 250;               // The target profits.
bool                bBuy_Stop = false;      // The signal for stopping buy order.
bool                bBuy_Pass = false;      // The signal for buy order.
string              sBuy_Signal = "NOSIG";  // The signal value for buy.
bool                bSell_Stop = false;     // The signal for stopping sell order.
bool                bSell_Pass = false;     // The signal for sell order.
string              sSell_Signal = "NOSIG"; // The signal value for sell.

// Set the predicted values directory.
string              DataDir = "FXStrategy"; // The data directory.
string              HLCOPredFileName = "";  // The predicted HLCO filename.
int                 HLCOPredFileHandle;     // The predicted HLCO file handle.
string              TrainDataFileName = ""; // The train data filename.
int                 TrainDataFileHandle;    // The train data file handle.

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

    // Compute the NQPR.
    ComputeNQPR();
    // Get the previous close.
    pClose = iClose(TPSymbol, PERIOD_D1, 1);
    // Get the prediction.
    HLCOPredFileName = TPSymbol + "_Pred.csv";
    HLCOPredFileHandle = FileOpen(DataDir + "//Preds//" + HLCOPredFileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
    // Get the value.
    predO = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
    // Close the file.
    FileClose(HLCOPredFileHandle);
    // Rectify the predicted value.
    predO = RectifyPred(pClose, predO);
    // Compute the QPLs.
    nSell_Pass = predO * NQPR[0];
    nBuy_Pass = predO / NQPR[0];

    // if (TimeHour(TimeLocal()) < 21)
    // {
    //     // Get the prediction.
    //     HLCOPredFileName = (tradeDay - 1) + "_" + TPSymbol + "_Pred.csv";
    //     HLCOPredFileHandle = FileOpen(DataDir + "//Preds//" + HLCOPredFileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
    //     // Get the value.
    //     predO = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
    //     // Close the file.
    //     FileClose(HLCOPredFileHandle);
    //     // Get the previous close.
    //     pClose = iClose(TPSymbol, PERIOD_D1, 1);
    //     // Rectify the predicted value.
    //     predO = RectifyPred(pClose, predO);
    //     // Compute the QPLs.
    //     nSell_Pass = predO * NQPR[0];
    //     nBuy_Pass = predO / NQPR[0];
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
            // Get the train data.
            TrainDataFileName = TPSymbol + "_Train.csv";
            FileDelete(DataDir + "//Train//" + TrainDataFileName, FILE_COMMON);
            ResetLastError();
            TrainDataFileHandle = FileOpen(DataDir + "//Train//" + TrainDataFileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
            // Write the label into the csv file.
            FileWrite(TrainDataFileHandle, "Day", "Open", "High", "Low", "Close", "QPL+1", "QPL+2", "QPL+3", "QPL+4", "QPL+5", "QPL+6", "QPL+7", "QPL+8",
                        "QPL+9", "QPL+10", "QPL+11", "QPL+12", "QPL+13", "QPL+14", "QPL+15", "QPL+16", "QPL+17", "QPL+18", "QPL+19", "QPL+20", "QPL+21", "QPL-1", "QPL-2",
                        "QPL-3", "QPL-4", "QPL-5", "QPL-6", "QPL-7", "QPL-8", "QPL-9", "QPL-10", "QPL-11", "QPL-12", "QPL-13", "QPL-14", "QPL-15", "QPL-16",
                        "QPL-17", "QPL-18", "QPL-19", "QPL-20", "QPL-21");
            // Compute the NQPR.
            ComputeNQPR();
            // Compute and store all the training data into the file.
            for (d = 0;  d < 10; d++)
            {
                // Compute each QPL.
                for (eL = 0; eL < 21; eL++)
                {
                    PQPL[eL] = DT_CL[d] * NQPR[eL];
                    NQPL[eL] = DT_CL[d] / NQPR[eL];
                }
                // Store the training data into the file.
                FileWrite(TrainDataFileHandle, (d + 1),
                            DT_OP[d], DT_HI[d], DT_LO[d], DT_CL[d],
                            PQPL[0], PQPL[1], PQPL[2], PQPL[3], PQPL[4],
                            PQPL[5], PQPL[6], PQPL[7], PQPL[8], PQPL[9],
                            PQPL[10], PQPL[11], PQPL[12], PQPL[13], PQPL[14],
                            PQPL[15], PQPL[16], PQPL[17], PQPL[18], PQPL[19], PQPL[20],
                            NQPL[0], NQPL[1], NQPL[2], NQPL[3], NQPL[4],
                            NQPL[5], NQPL[6], NQPL[7], NQPL[8], NQPL[9],
                            NQPL[10], NQPL[11], NQPL[12], NQPL[13], NQPL[14],
                            NQPL[15], NQPL[16], NQPL[17], NQPL[18], NQPL[19], NQPL[20]);
            }
            // Close file.
            FileClose(TrainDataFileHandle);

            // Get the prediction.
            HLCOPredFileName = TPSymbol + "_Pred.csv";
            HLCOPredFileHandle = FileOpen(DataDir + "//Preds//" + HLCOPredFileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
            // Get the value.
            predO = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
            // Close the file.
            FileClose(HLCOPredFileHandle);
            // Get the previous close.
            pClose = iClose(TPSymbol, PERIOD_D1, 0);
            // Rectify the predicted value.
            predO = RectifyPred(pClose, predO);
            // Compute the QPLs.
            nSell_Pass = predO * NQPR[0];
            nBuy_Pass = predO / NQPR[0];

            // // Get the prediction.
            // HLCOPredFileName = (tradeDay - 1) + "_" + TPSymbol + "_Pred.csv";
            // HLCOPredFileHandle = FileOpen(DataDir + "//Preds//" + HLCOPredFileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
            // // Get the value.
            // predO = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
            // // Close the file.
            // FileClose(HLCOPredFileHandle);
            // // Get the previous close.
            // pClose = iClose(TPSymbol, PERIOD_D1, 0);
            // // Rectify the predicted value.
            // predO = RectifyPred(pClose, predO);
            // // Compute the QPLs.
            // nSell_Pass = predO * NQPR[0];
            // nBuy_Pass = predO / NQPR[0];
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
double RectifyPred(double cClose, double x)
{
    // Get the rectified prediction value.
    return (cClose + x) / 2;
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

// Create the function to compute the NQPR.
void ComputeNQPR()
{
    // Compute the 21 K values.
    for (eL = 0; eL < maxEL; eL++)
    {
        K[eL] = MathPow((1.1924 + (33.2383 * eL) + 56.2169 * MathPow(eL, 2)) / (1 + (43.6196 * eL)), p3);
    }

    // Get the total data time series.
    TSsize = 0;
    while(iTime(TPSymbol, PERIOD_D1, TSsize) > 0 && (TSsize < maxTS))
    {
        TSsize++;
    }

    // Get all the high, low, close and open.
    for (d = 1; d < TSsize; d++)
    {
        DT_OP[d - 1] = iOpen(TPSymbol, PERIOD_D1, d);
        DT_HI[d - 1] = iHigh(TPSymbol, PERIOD_D1, d);
        DT_LO[d - 1] = iLow(TPSymbol, PERIOD_D1, d);
        DT_CL[d - 1] = iClose(TPSymbol, PERIOD_D1, d);
        DT_RT[d - 1] = 1;
    }

    // Calculate return.
    for (d = 0; d < (TSsize - 2); d++)
    {
        if (DT_CL[d + 1] > 0)
        {
            DT_RT[d] = DT_CL[d] / DT_CL[d + 1];
        }
        else
        {
            DT_RT[d] = 1;
        }
    }

    // Get the maximum time series number of the return.
    maxRno = TSsize - 2;

    // Calculate mean.
    mu = 0;
    for (d = 0; d < maxRno; d++)
    {
        mu = mu + DT_RT[d];
    }
    mu = mu / maxRno;

    // Calculate standard deviation.
    sigma = 0;
    for (d = 0; d < maxRno; d++)
    {
        sigma = sigma + MathPow((DT_RT[d] - mu), 2);
    }
    sigma = sqrt((sigma / maxRno));

    // Calculate dr for each return in the PDF of return.
    dr = 3 * sigma / 50;

    // Compute the PDF of the returns of simulate wave function of the returns.
    auxR = 0;
    // Reset all the Q[] first.
    for (nQ = 0; nQ < 100; nQ++)
    {
        Q[nQ] = 0;
    }

    // Loop over maxRno to get the distribution.
    tQno = 0;
    for (nR = 0; nR < maxRno; nR++)
    {
        bFound = false;
        nQ = 0;
        // Get the start position of the wave function.
        auxR = 1 - (dr * 50);
        // Get the total number of the returns in each range of each segment of the wave function.
        while (!bFound && (nQ < 100))
        {
            if ((DT_RT[nR] > auxR) && (DT_RT[nR] <= (auxR + dr)))
            {
                Q[nQ]++;
                tQno++;
                bFound = true;
            }
            else
            {
                nQ++;
                auxR = auxR + dr;
            }
        } 
    }

    // Get the start position of the wave function.
    auxR = 1 - (dr * 50);
    // Normalize the wave function.
    for (nQ = 0; nQ < 100; nQ++)
    {
        r[nQ] = auxR;
        NQ[nQ] = Q[nQ] / tQno;
        auxR = auxR + dr;
    }

    // Find the max value and its corresponding return in the wave function.
    maxQ = 0;
    maxQno = 0;
    for (nQ = 0; nQ < 100; nQ++)
    {
        if (NQ[nQ] > maxQ)
        {
            maxQ = NQ[nQ];
            maxQno = nQ;
        }
    }

    // Compute the lambda value.
    r0 = r[maxQno] - (dr / 2);
    // Get the r+1's return.
    r1 = r0 + dr;
    // Get the r-1's return.
    rn1 = r0 - dr;
    // Compute the numerator of the lambda.
    Lup = (pow(rn1, 2) * NQ[maxQno - 1]) - (pow(r1, 2) * NQ[maxQno + 1]);
    // Compute the denominator of the lambda.
    Ldw = (pow(rn1, 4) * NQ[maxQno - 1]) - (pow(r1, 4) * NQ[maxQno + 1]);
    // Compute the lambda.
    L = MathAbs(Lup / Ldw);

    // Use the Cardano's method to compute the 21  Quantum Finanace Energy Level.
    for (eL = 0; eL < 21; eL++)
    {
        // Compute the coefficients of the depressed cubic equation.
        p = -1 * pow((2 * eL + 1), 2);
        q = -1 * L * pow((2 * eL + 1), 3) * pow(K[eL], 3);

        // Apply Cardano's method to find the real root of the depressed cubic equation.
        u = MathPow((-0.5 * q + MathSqrt(((q * q / 4.0) + (p * p * p / 27.0)))), p3);
        v = MathPow((-0.5 * q - MathSqrt(((q * q / 4.0) + (p * p * p / 27.0)))), p3);

        // Store the QFEL.
        QFEL[eL] = u + v;
    }

    // Evaluate all QPR values.
    for (eL = 0; eL < 21; eL++)
    {
        // Compute the QPR.
        QPR[eL] = QFEL[eL] / QFEL[0];
        // Compute the NQPR.
        NQPR[eL] = 1 + 0.21 * sigma * QPR[eL];
    }
}