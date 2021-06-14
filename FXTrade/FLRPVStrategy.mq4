//+------------------------------------------------------------------+
//|                                                FLRPVStrategy.mq4 |
//|                Copyright 2021, J. Lee, J. Huang, O. Lin, V. Guo. |
//|                                                  http://qffc.org |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, J. Lee, J. Huang, O. Lin, V. Guo."
#property link      "http://qffc.org"
#property version   "1.00"
#property strict

// Define the global variables for QPLs.
int      maxEL = 21;             // The maximum energy level.
int      maxTS = 2048;           // The maximun time series for QPL and RPV.
double   p3 = 1.0 / 3.0;         // The power of the K value.
// Declare the variables to compute the NQPR.
int      eL, d, TSsize, nQ, maxQno, nR, maxRno, tQno;
double   auxR, maxQ, r0, r1, rn1;
double   dr, Lup, Ldw, L, mu, sigma;
bool     bFound;
// Variables used in Cardano's Method.
double   p, q, u, v;   
// Declare time series array.
double   DT_OP[2048];
double   DT_HI[2048];
double   DT_LO[2048];
double   DT_CL[2048];
double   DT_RT[2048];
// Declare array for Quantum Price Wavefunction.
double   Q[100];
double   NQ[100];
double   r[100];   
// Declare array for NQPR related arrays.
double   QFEL[21];               // The QFEL for the current product.
double   QPR[21];                // The QPR for the current product.
double   NQPR[21];               // The NQPR for the current product.
double   QPLs[42];               // The QPLs for the current product.
double   PQPL[21];               // The positive QPL for the current product.
double   NQPL[21];               // The negative QPL for the current product.
double   K[21];                  // The K value for the current product.
// Set the variables to store the boundaries of the RPV.
double   RPVBound[5];            // The array for RPV Boundaries.
double   RPV[2048];              // The array for RPV values.
double   Mean;                   // The mean of the RPV.
double   Sigma;                  // The sigma of the RPV.

// Define the file directory.
string   DataDir = "FXStrategy"; // The data directory.
string   DataFileName = "";      // The data filename.
int      DataFileHandle;         // The data file handle.
string   HLCOPredFileName = "";  // The predicted HLCO filename.
int      HLCOPredFileHandle;     // The predicted HLCO file handle.
string   TrainDataFileName = ""; // The train data filename.
int      TrainDataFileHandle;    // The train data file handle.

// Define the product symbol variables.
string   TPSymbol = "";          // The current product symbol.

// Set the trade control variables.
int      pHH = 0;                // The previous hour.
int      cHH = 0;                // The current hour.
double   TPLot = 1.0;            // The total transaction volume.
string   TPMagic = 88888;        // The magic value of the product.
double   cPrice = 0;             // The current price of the product.
double   sPrice = 0;             // The string of the current price of the product.
double   cOpen = 0;              // The current open for the current product.
double   cRPV = 0;               // The current relative price volatility.
double   cRSI = 0;               // The current relative strength index.
double   nBuy_Pass = 30.0;       // The threshold of over sell for RSI.
double   nSell_Pass = 70.0;      // The threshold of over buy for RSI.
double   iRPV[3];                // The implication value of the RPV.
double   icRSI[3];               // The implication value of the current RSI.
double   DF = 0;                 // The value of the defuzzification.
int      sl = 150;               // The stop loss.
int      tp = 300;               // The target profits.
double   predH = 0;              // The predicted high value.
double   predL = 0;              // The predicted low value.
double   predO = 0;              // The predicted open value.
double   predC = 0;              // The predicted close value.
bool     bBuy_Stop = false;      // The signal for stopping buy order.
bool     bBuy_Pass = false;      // The signal for buy order.
string   sBuy_Signal = "NOSIG";  // The signal value for buy.
bool     bSell_Stop = false;     // The signal for stopping sell order.
bool     bSell_Pass = false;     // The signal for sell order.
string   sSell_Signal = "NOSIG"; // The signal value for sell.

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
   
   // Create the file handle to store and read the NQPR.
   DataFileName = TPSymbol + "_DATA.csv";
   FileDelete(DataDir + "//DATA//" + DataFileName, FILE_COMMON);
   ResetLastError();
   DataFileHandle = FileOpen(DataDir + "//DATA//" + DataFileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
   // Compute the NQPR and RPV.
   ComputeStrategyData();
   DataFileHandle = FileOpen(DataDir + "//DATA//" + DataFileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
   // Get the current open.
   cOpen = iOpen(TPSymbol, PERIOD_D1, 0);
   // Read the NQPR.
   for (int i = 0; i < maxEL; i++)
   {
      NQPR[i] = StringToDouble(FileReadString(DataFileHandle, 1));
      // Compute the QPL.
      PQPL[i] = cOpen * NQPR[i];
      NQPL[i] = cOpen / NQPR[i];
   }
   // Get the mean and sigma of the RPV.
   Mean = StringToDouble(FileReadString(DataFileHandle, 1));
   Sigma = StringToDouble(FileReadString(DataFileHandle, 1));
   // Store all QPLs.
   for (int i = maxEL - 1; i >= 0; i--)
   {
      QPLs[(maxEL - 1) - i] = PQPL[i];
   }
   for (int i = 0; i < maxEL; i++)
   {
      QPLs[i + maxEL] = NQPL[i];
   }
   // Close the file.
   FileClose(DataFileHandle);
   
   // Create the file handle to read the predicted values.
   HLCOPredFileName = TPSymbol + "_Pred.csv";
   HLCOPredFileHandle = FileOpen(DataDir + "//PREDs//" + HLCOPredFileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
   // Get the value.
   predO = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
   predH = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
   predL = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
   predC = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
   // Close the file.
   FileClose(HLCOPredFileHandle);
   // Rectify the predicted value according to the current open.
   predH = RectifyPredHL(RectifyPred(cOpen, predO, predH));
   predL = RectifyPredHL(RectifyPred(cOpen, predO, predL));
   predC = RectifyPred(cOpen, predO, predC);
   predO = RectifyPred(cOpen, predO, predO);
   
   // Get the RPV boundaries.
   RPVBound[0] = RPVNormalDistribution(Mean, Sigma, (Mean - 3 * Sigma));
   RPVBound[4] = RPVNormalDistribution(Mean, Sigma, (Mean - Sigma));
   RPVBound[2] = (RPVBound[0] + RPVBound[4]) / 2.0;
   RPVBound[1] = (RPVBound[0] + RPVBound[2]) / 2.0;
   RPVBound[3] = (RPVBound[2] + RPVBound[4]) / 2.0;
   
   // Print all the QPLs.
   for (int i = 0; i < 2 * maxEL; i++)
   {
      Print("The ", (i + 1), " QPL is ", QPLs[i]);
   }
   
   // Print all the RPV boundaries.
   for (int i = 0; i < 5; i++)
   {
      Print("The ", (i + 1), " RPV boundary is ", RPVBound[i]);
   }
   
   // Compute the current RPV.
   cRPV = MathAbs(predC - predO) / (predH - predL);
   // Convert the current RPV to be the value in normal distribution.
   cRPV = RPVNormalDistribution(Mean, Sigma, cRPV);
   // Compute the fuzzy variables implication for RPV.
   iRPV[0] = RPVImplication("MUST");
   iRPV[1] = RPVImplication("MAY");
   iRPV[2] = RPVImplication("NO");

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
      
      // Create the file handle to store and read the NQPR.
      DataFileName = TPSymbol + "_DATA.csv";
      FileDelete(DataDir + "//DATA//" + DataFileName, FILE_COMMON);
      ResetLastError();
      DataFileHandle = FileOpen(DataDir + "//DATA//" + DataFileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
      // Compute the NQPR and RPV.
      ComputeStrategyData();
      DataFileHandle = FileOpen(DataDir + "//DATA//" + DataFileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
      // Get the current open.
      cOpen = iOpen(TPSymbol, PERIOD_D1, 0);
      // Read the NQPR.
      for (int i = 0; i < maxEL; i++)
      {
         NQPR[i] = StringToDouble(FileReadString(DataFileHandle, 1));
         // Compute the QPL.
         PQPL[i] = cOpen * NQPR[i];
         NQPL[i] = cOpen / NQPR[i];
      }
      // Get the mean and sigma of the RPV.
      Mean = StringToDouble(FileReadString(DataFileHandle, 1));
      Sigma = StringToDouble(FileReadString(DataFileHandle, 1));
      // Store all QPLs.
      for (int i = maxEL - 1; i >= 0; i--)
      {
         QPLs[(maxEL - 1) - i] = PQPL[i];
      }
      for (int i = 0; i < maxEL; i++)
      {
         QPLs[i + maxEL] = NQPL[i];
      }
      // Close the file.
      FileClose(DataFileHandle);
   
      // Create the file handle to read the predicted values.
      HLCOPredFileName = TPSymbol + "_Pred.csv";
      HLCOPredFileHandle = FileOpen(DataDir + "//PREDs//" + HLCOPredFileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
      // Get the value.
      predO = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
      predH = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
      predL = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
      predC = StringToDouble(FileReadString(HLCOPredFileHandle, 1));
      // Close the file.
      FileClose(HLCOPredFileHandle);
      // Rectify the predicted value according to the current open.
      predH = RectifyPredHL(RectifyPred(cOpen, predO, predH));
      predL = RectifyPredHL(RectifyPred(cOpen, predO, predL));
      predC = RectifyPred(cOpen, predO, predC);
      predO = RectifyPred(cOpen, predO, predO);
   
      // Get the RPV boundaries.
      RPVBound[0] = RPVNormalDistribution(Mean, Sigma, (Mean - 3 * Sigma));
      RPVBound[4] = RPVNormalDistribution(Mean, Sigma, (Mean - Sigma));
      RPVBound[2] = (RPVBound[0] + RPVBound[4]) / 2.0;
      RPVBound[1] = (RPVBound[0] + RPVBound[2]) / 2.0;
      RPVBound[3] = (RPVBound[2] + RPVBound[4]) / 2.0;
   
      // Print all the QPLs.
      for (int i = 0; i < 2 * maxEL; i++)
      {
         Print("The ", (i + 1), " QPL is ", QPLs[i]);
      }
   
      // Print all the RPV boundaries.
      for (int i = 0; i < 5; i++)
      {
         Print("The ", (i + 1), " RPV boundary is ", RPVBound[i]);
      }
   
      // Compute the current RPV.
      cRPV = MathAbs(predC - predO) / (predH - predL);
      // Convert the current RPV to be the value in normal distribution.
      cRPV = RPVNormalDistribution(Mean, Sigma, cRPV);
      // Compute the fuzzy variables implication for RPV.
      iRPV[0] = RPVImplication("MUST");
      iRPV[1] = RPVImplication("MAY");
      iRPV[2] = RPVImplication("NO");
   }
   // Give the hint.
   Print("Pred High = ", predH, " Pred Low = ", predL, " Pred Close = ", predC, " Pred Open = ", predO);
   Print("cRPV = ", cRPV, " Implication (MUST, MAY, NO) = (", iRPV[0], ", ", iRPV[1], ", ", iRPV[2], ")");

   // Check whether reach the sleep time.
   if (TimeHour(TimeLocal()) >= 21)
   {
      bBuy_Stop = true;
      sBuy_Signal = "SLEEPING";
      bSell_Stop = true;
      sSell_Signal = "SLEEPING";
      
      if (TimeHour(TimeLocal()) == 21)
      {
         // Create the file handle to store the train data for next day.
         TrainDataFileName = TPSymbol + "_Train.csv";
         FileDelete(DataDir + "//TRAIN//" + TrainDataFileName, FILE_COMMON);
         ResetLastError();
         TrainDataFileHandle = FileOpen(DataDir + "//TRAIN//" + TrainDataFileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
      
         // Compute the 21 K values.
         for (eL = 0; eL < maxEL; eL++)
         {
            K[eL] = MathPow((1.1924 + (33.2383 * eL) + (56.2169 * eL * eL)) / (1 + (43.6106 * eL)), p3); 
         }
         // Get the total data time series.
         TSsize = 0;
         while (iTime(TPSymbol, PERIOD_D1, TSsize) > 0 && (TSsize < maxTS))
         {
            TSsize++;
         }  
      
         // Get all the High, Low, Close and Open training data.
         for (d = 0; d < (TSsize - 1); d++)
         {
            DT_OP[d] = iOpen(TPSymbol, PERIOD_D1, d);
            DT_HI[d] = iHigh(TPSymbol, PERIOD_D1, d);
            DT_LO[d] = iLow(TPSymbol, PERIOD_D1, d);
            DT_CL[d] = iClose(TPSymbol, PERIOD_D1, d);
            DT_RT[d] = 1;
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
            sigma = sigma + (DT_RT[d] - mu) * (DT_RT[d] - mu);
         }
         sigma = sqrt((sigma / maxRno));      
      
         // Calculate dr for each return in the PDF of return.
         dr = 3 * sigma / 50;
      
         // Compute the PDF of the returns to simulate wave function of the returns.
         auxR = 0;
         // Loop over all r from (r-50*dr) to (r+50*dr) and get the distribution function.
         // Reset all the Q[] first.
         for (nQ = 0; nQ < 100; nQ++)
         {
            Q[nQ] = 0;
         }
      
         // Loop over the maxRno to get the distribution.
         tQno = 0;
         for (nR = 0; nR < maxRno; nR++)
         {
            bFound = False;
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
                  bFound = True;
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
      
         // Use the Cardano's to compute the 21 Quantum Finance Energy Level. 
         for (eL = 0; eL < 21; eL++)
         {
            // Compute the coefficients of the depressed cubic equation.
            p = -1 * pow((2 * eL + 1), 2);
            q = -1 * L * pow((2 * eL + 1), 3) * pow(K[eL], 3);
         
            // Apply Cardano's Method to find the real root of the depressed cubic equation.
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
      
         // Write the label into the csv file.
         FileWrite(TrainDataFileHandle, "Day", "Open", "High", "Low", "Close", "QPL+1", "QPL+2", "QPL+3", "QPL+4", "QPL+5", "QPL+6", "QPL+7", "QPL+8",
                  "QPL+9", "QPL+10", "QPL+11", "QPL+12", "QPL+13", "QPL+14", "QPL+15", "QPL+16", "QPL+17", "QPL+18", "QPL+19", "QPL+20", "QPL+21", "QPL-1", "QPL-2",
                  "QPL-3", "QPL-4", "QPL-5", "QPL-6", "QPL-7", "QPL-8", "QPL-9", "QPL-10", "QPL-11", "QPL-12", "QPL-13", "QPL-14", "QPL-15", "QPL-16",
                  "QPL-17", "QPL-18", "QPL-19", "QPL-20", "QPL-21");
      
         // Compute and store all the training data into the file.
         for (d = 0; d < 10; d++)
         {
            // Compute each QPL.
            for (eL = 0; eL < 21; eL++)
            {
               PQPL[eL] = DT_OP[d] * NQPR[eL];
               NQPL[eL] = DT_OP[d] / NQPR[eL];
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
      }
   }
   // Get the currrent RSI.
   cRSI = iRSI(TPSymbol, PERIOD_H1, 14, PRICE_CLOSE, 0);
   // Compute the fuzzy variables implication for current RSI.
   icRSI[0] = RSIImplication("BUY");
   icRSI[1] = RSIImplication("NO");
   icRSI[2] = RSIImplication("SELL");
   
   // Compute the defuzzification.
   DF = Defuzzificate();
   
   // Give the hint.
   Print("cRSI = ", cRSI, " Implication (BUY, NO, SELL) = (", icRSI[0], ", ", icRSI[1], ", ", icRSI[2], ")");
   Print("Defuzzification Value ", DF);
   
   // Prevent more sell order if there are some outstanding sell orders.
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
   // Check whether reach the stop loss.
   //if (true)
   //{
   //   Print("Stop Loss");
   //}
   // Check whether to make the buy order.
   if (!bBuy_Stop && !bBuy_Pass && (DF <= nBuy_Pass))
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
   Print(TPSymbol, " : Price = ", sPrice, " RSI = ", cRSI, " DF = ", DF, " BUYSIGNAL = ", sBuy_Signal);
   
   // Get the buy price.
   cPrice = Bid;
   sPrice = DoubleToString(cPrice, 5);
   // Check whether reach the target profits.
   //if (true)
   //{
   //   Print("Target Profits");
   //}
   // Check whether to make the sell order.
   if (!bSell_Stop && !bSell_Pass && (DF >= nSell_Pass))
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
      }
   }
   // Print the current status.
   Print(TPSymbol, " : Price = ", sPrice, " RSI = ", cRSI, " DF = ", DF, " SELLSIGNAL = ", sSell_Signal);
   
   // Update the previous hour.
   pHH = cHH;
   // Print the order status.
   Print("Buy Order : ", IntegerToString(nActiveBuyOrder(), 10));
   Print("Sell Order : ", IntegerToString(nActiveSellOrder(), 10));
   // Sleep for 15 seconds.
   Sleep(15000);
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

// Create the function to defuzzificate the fuzzy logic.
double Defuzzificate()
{
   // Create the array to store all the implication value of the fuzzy rules.
   double iRules[9];
   // Create the array to store all the weight for each rules.
   double w[3];
   w[0] = (nBuy_Pass + 5) / 2;           // The weight for buy.
   w[1] = (nBuy_Pass + nSell_Pass) / 2; // The weight for no.
   w[2] = (100 + (nSell_Pass - 5)) / 2;  // The weight for sell.
   // Compute the fuzzy rules.
   iRules[0] = MathMin(iRPV[0], icRSI[0]);
   iRules[1] = MathMin(iRPV[1], icRSI[0]);
   iRules[2] = MathMin(iRPV[2], icRSI[0]);
   iRules[3] = MathMin(iRPV[0], icRSI[1]);
   iRules[4] = MathMin(iRPV[1], icRSI[1]);
   iRules[5] = MathMin(iRPV[2], icRSI[1]);
   iRules[6] = MathMin(iRPV[0], icRSI[2]);
   iRules[7] = MathMin(iRPV[1], icRSI[2]);
   iRules[8] = MathMin(iRPV[2], icRSI[2]);
   // Defuzzification.
   double dfw = (iRules[0] * w[0] + iRules[1] * w[0] + iRules[2] * w[1] + iRules[3] * w[1] + iRules[4] * w[1] + iRules[5] * w[1] + iRules[6] * w[2] + iRules[7] * w[2] + iRules[8] * w[1]);
   double df = (iRules[0] + iRules[1] + iRules[2] + iRules[3] + iRules[4] + iRules[5] + iRules[6] + iRules[7] + iRules[8]);
   // Return the result.
   return dfw / df;
}

// Create the function for RSI's implication computation.
double RSIImplication(string mode)
{
   // Compute the implication value of the different modes.
   if (mode == "NO")
   {
      if (cRSI >= (nBuy_Pass + 5) && cRSI <= (nSell_Pass - 5))
      {
         return 1.0;
      }
      else if (cRSI <= nBuy_Pass || cRSI >= nSell_Pass)
      {
         return 0.0;
      }
      else if (cRSI > nBuy_Pass && cRSI < (nBuy_Pass + 5))
      {
         // Compute the slope.
         double k = (1 - 0) / ((nBuy_Pass + 5) - nBuy_Pass);
         // Compute the bias.
         double b = -k * nBuy_Pass;
         return k * cRSI + b;
      }
      else
      {
         // Compute the slope.
         double k = (0 - 1) / (nSell_Pass - (nSell_Pass - 5));
         // Compute the bias.
         double b = -k * nSell_Pass;
         return k * cRSI + b;
      }
   }
   else if (mode == "BUY")
   {
      if (cRSI <= nBuy_Pass)
      {
         return 1.0;
      }
      else if (cRSI >= (nBuy_Pass + 5))
      {
         return 0.0;
      }
      else
      {
         // Compute the slope.
         double k = (0 - 1) / ((nBuy_Pass + 5) - nBuy_Pass);
         // Compute the bias.
         double b = -k * (nBuy_Pass + 5);
         return k * cRSI + b;
      }
   }
   else
   {
      if (cRSI >= nSell_Pass)
      {
         return 1.0;
      }
      else if (cRSI <= (nSell_Pass - 5))
      {
         return 0.0;
      }
      else
      {
         // Compute the slope.
         double k = (1 - 0) / (nSell_Pass - (nSell_Pass - 5));
         // Compute the bias.
         double b = -k * (nSell_Pass - 5);
         return k * cRSI + b;
      }
   }
}

// Create the function for RPV's implication computation.
double RPVImplication(string mode)
{
   // Compute the implication value of the different modes.
   if (mode == "NO")
   {
      if (cRPV <= RPVBound[3])
      {
         return 0.0;
      }
      else if (cRPV > RPVBound[3] && cRPV < RPVBound[4])
      {
         // Compuet the slope.
         double k = (1 - 0) / (RPVBound[4] - RPVBound[3]);
         // Compute the bias.
         double b = -k * RPVBound[3];
         return k * cRPV + b;
      }
      else
      {
         return 1.0;
      }
   }
   else if (mode == "MAY")
   {
      if (cRPV <= RPVBound[0] || cRPV >= RPVBound[4])
      {
         return 0.0;
      }
      else if (cRPV > RPVBound[0] && cRPV < RPVBound[1])
      {
         // Compute the slope.
         double k = (1 - 0) / (RPVBound[1] - RPVBound[0]);
         // Compute the bias.
         double b = -k * RPVBound[0];
         return k * cRPV + b;
      }
      else if (cRPV >= RPVBound[1] && cRPV <= RPVBound[3])
      {
         return 1.0;
      }
      else
      {
         // Compute the slope.
         double k = (0 - 1) / (RPVBound[4] - RPVBound[3]);
         // Compute the bias.
         double b = -k * RPVBound[4];
         return k * cRPV + b;
      }
   }
   else
   {
      if (cRPV <= RPVBound[0])
      {
         return 1.0;
      }
      else if (cRPV >= RPVBound[1])
      {
         return 0.0;
      }
      else
      {
         // Compute the slope.
         double k = (0 - 1) / (RPVBound[1] - RPVBound[0]);
         // Compute the bias.
         double b = -k * RPVBound[1];
         return k * cRPV + b;
      }
   }
}

// Create the function to compute the normal distribution.
double RPVNormalDistribution(double Mean, double Sigma, double x)
{
   // Create the variables to store the value.
   double fx = 0;
   // Compute the fx.
   fx = (1.0 / (Sigma * MathSqrt(2.0 * 3.1415926535))) * MathExp((-1.0 / 2.0) * MathPow(((x - Mean) / Sigma), 2.0));
   // Return the fx.
   return fx;
}

// Create the function to compare the H and L with the QPLs.
double RectifyPredHL(double x)
{
   // Set the min value.
   double min = MathAbs(x - QPLs[0]);
   // Set the number to store the index of the QPLs.
   int index = 0;
   // Get the minimum value.
   for (int i = 1; i < 2 * maxEL; i++)
   {
      if (min > (MathAbs(x - QPLs[i])))
      {
         min = MathAbs(x - QPLs[i]);
         index = i;
      }
   }
   // Check whether the minimum loss is smaller than the boundary.
   if (min < 0.1)
   {
      return QPLs[index];
   }
   else
   {
      return x;
   }
}

// Create the function to rectify the prediction.
double RectifyPred(double cOpen, double pOpen, double x)
{
   // Get the loss between the current open and predicted open.
   double loss = pOpen - cOpen;
   // Get the rectified prediction value.
   return (x - loss);
}

// Create the function to compute the NQPR and RPV.
void ComputeStrategyData()
{  
   // Compute the 21 K values.
   for (eL = 0; eL < maxEL; eL++)
   {
      K[eL] = MathPow((1.1924 + (33.2383 * eL) + 56.2169 * MathPow(eL, 2)) / (1 + (43.6106 * eL)), p3);
   }
   
   // Get the total data time series.
   TSsize = 0;
   while (iTime(TPSymbol, PERIOD_D1, TSsize) > 0 && (TSsize < maxTS))
   {
      TSsize++;
   }
   
   // Get all the High, Low, Close and Open training data.
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
   
   // Compute the PDF of the returns to simulate wave function of the returns.
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
   
   // Use the Cardano's to compute the 21 Quantum Finance Energy Level.
   for (eL = 0; eL < 21; eL++)
   {
      // Compute the coefficients of the depressed cubic equation.
      p = -1 * pow((2 * eL + 1), 2);
      q = -1 * L * pow((2 * eL + 1), 3) * pow(K[eL], 3);
      
      // Apply Cardano's Method to find the real root of the depressed cubic equation.
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
      // Write the values of the NQPR into a file.
      FileWrite(DataFileHandle, NQPR[eL]);
   }
   
   // Compute the RPVs.
   for (d = 1; d < TSsize; d++)
   {
      RPV[d - 1] = MathAbs(DT_CL[d - 1] - DT_OP[d - 1]) / (DT_HI[d - 1] - DT_LO[d - 1]);
   }
   
   // Compute the mean.
   Mean = 0;
   for (d = 0; d < maxRno; d++)
   {
      Mean = Mean + RPV[d];
   }
   Mean = Mean / maxRno;
   
   // Compute the standard deviation.
   Sigma = 0;
   for (d = 0; d < maxRno; d++)
   {
      Sigma = Sigma + MathPow((RPV[d] - Mean), 2);
   }
   Sigma = sqrt((Sigma / maxRno));
   
   // Write the values of the RPVs.
   FileWrite(DataFileHandle, Mean);
   FileWrite(DataFileHandle, Sigma);
   
   // Close the file.
   FileClose(DataFileHandle);
}