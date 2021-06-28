//+------------------------------------------------------------------+
//|                                                  FXTrainData.mq4 |
//|                        Copyright 2021, J. Huang, O. Lin, V. Guo. |
//|                                                  http://qffc.org |
//+------------------------------------------------------------------+

#property copyright "Copyright © 2021, J. Huang, O. Lin, V. Guo."
#property link      "http://qffc.org"

// Define the file directory.
string   QP_Directory = "FXTrainData"; // QPL Directory.
string   Currency_QPL_FileName = "";   //File name for each currency.
int      Currency_QPL_FileHandle;      //File Handl for each currency.

// Define the global variables.
int      maxELevel = 21;               // Max Energy Level.      
int      maxTP = 10;                   // Max no of Financial Product.
int      maxTS = 2048;                 // Max no of Time Series Record.
int      nTP = 0;                      // The number of the product symbol.
double   p3 = 1.0 / 3.0;               // The power of the K value.

//Define the timing variables.
uint     stime = 0;
uint     etime = 0;
uint     Gstime = 0;
uint     Getime = 0;
uint     tlapse = 0;
uint     Gtlapse = 0;

// Define the product symbol variables.
string   TPSymbol = "";
string   TP_Code[10] = {"AUDNZD", "CADCHF", "EURAUD", "EURCAD", "EURUSD", "GBPUSD", "USDCAD", "USDCHF", "USDHKD", "GBPCHF"};
int      TP_No[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
int      TP_nD[10] = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5};

// Deinit.
void OnDeinit(const int reason){}

// Initialize.
int OnInit()
{
   int      eL, d, TSsize, nQ, maxQno, nR, maxRno, tQno;
   double   auxR, maxQ, r0, r1, rn1;
   double   dr, Lup, Ldw, L, mu, sigma; 
   bool     bFound;
   
   // Variables used in Cardano's Method.
   double   p, q, u, v;
         
   // Declare time series array.
   int      DT_YY[2048];
   int      DT_MM[2048];
   int      DT_DD[2048];
   double   DT_OP[2048];
   double   DT_HI[2048];
   double   DT_LO[2048];
   double   DT_CL[2048];
   double   DT_VL[2048];
   double   DT_RT[2048];
      
   // Declare array for Quantum Price Wavefunction.
   double   Q[100];           // Quantum Price Wavefunction.
   double   NQ[100];          // Normalized Q[].
   double   r[100];           // r no.
   
   // Declare array for QPL related arrays.
   double   QFEL[21];         // QFEL for each FP.
   double   QPR[21];          // QPR  for each FP.
   double   NQPR[21];         // NQPR for each FP.
   double   K[21];            // K values in QP Schrodinger Eqt.
   double   ALL_Pos_QPL[21];  // The positive 21 QPLs.
   double   ALL_Neg_QPL[21];  // The negative 21 QPLs.
    
   // Set global start time.
   Gstime = GetTickCount();
   
   // Compute the 21 K values.
   Print("Printout ALL K values K0 .. K20 for first 20 Energy Levels");
   for (eL = 0; eL < 21; eL++)
   {
      K[eL] = MathPow((1.1924 + (33.2383 * eL) + (56.2169 * eL * eL)) / (1 + (43.6106 * eL)), p3);
      Print("Energy Level ", eL, " K", eL, " = ", K[eL]);   
   }
   
   // Compute the training data for each product.
   for (nTP = 0; nTP < maxTP; nTP++)
   {
      TPSymbol = TP_Code[nTP];   // Get TP Symbol.
      stime = GetTickCount();    // Get timer.

      // Create the file to store the training data.
      Currency_QPL_FileName = TP_Code[nTP] + "_Data.csv";   
      FileDelete(QP_Directory + "//" + Currency_QPL_FileName, FILE_COMMON);
      ResetLastError();
      Currency_QPL_FileHandle = FileOpen(QP_Directory + "//" + Currency_QPL_FileName, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_CSV, ',');
      
      // Get the total data time series.
      TSsize = 0;
      while (iTime(TPSymbol, PERIOD_D1, TSsize) > 0 && (TSsize < maxTS))
      {
         TSsize++;
      }  
      
      // Get all the High, Low, Close and Open training data.
      for (d = 1; d < TSsize; d++)
      {
          DT_YY[d-1] = TimeYear(iTime(TPSymbol, PERIOD_D1, d));     
          DT_MM[d-1] = TimeMonth(iTime(TPSymbol, PERIOD_D1, d));     
          DT_DD[d-1] = TimeDay(iTime(TPSymbol, PERIOD_D1, d));     
          DT_OP[d-1] = iOpen(TPSymbol, PERIOD_D1, d);
          DT_HI[d-1] = iHigh(TPSymbol, PERIOD_D1, d);
          DT_LO[d-1] = iLow(TPSymbol, PERIOD_D1, d);
          DT_CL[d-1] = iClose(TPSymbol, PERIOD_D1, d);
          DT_VL[d-1] = iVolume(TPSymbol, PERIOD_D1, d);
          DT_RT[d-1] = 1;
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
      FileWrite(Currency_QPL_FileHandle, "Date", "Open", "High", "Low", "Close", "QPL+1", "QPL+2", "QPL+3", "QPL+4", "QPL+5", "QPL+6", "QPL+7", "QPL+8",
      "QPL+9", "QPL+10", "QPL+11", "QPL+12", "QPL+13", "QPL+14", "QPL+15", "QPL+16", "QPL+17", "QPL+18", "QPL+19", "QPL+20", "QPL+21", "QPL-1", "QPL-2",
      "QPL-3", "QPL-4", "QPL-5", "QPL-6", "QPL-7", "QPL-8", "QPL-9", "QPL-10", "QPL-11", "QPL-12", "QPL-13", "QPL-14", "QPL-15", "QPL-16",
      "QPL-17", "QPL-18", "QPL-19", "QPL-20", "QPL-21");
      //FileWrite(Currency_QPL_FileHandle, "Date", "Open", "High", "Low", "Close", "Volume");
      
      // Store the data into the csv.
      for (d = /*102*/(TSsize - 2); d >= /*11*/0; d--)
      {
         // Compute each QPL.
         for (eL = 0; eL < 21; eL++)
         {
            ALL_Pos_QPL[eL] = DT_OP[d] * NQPR[eL];
            ALL_Neg_QPL[eL] = DT_OP[d] / NQPR[eL];
         }
         // Store the training data into the file.
         FileWrite(Currency_QPL_FileHandle, IntegerToString(DT_YY[d]) + "/" + IntegerToString(DT_MM[d]) + "/" + IntegerToString(DT_DD[d]),
             DT_OP[d], DT_HI[d], DT_LO[d], DT_CL[d],
             ALL_Pos_QPL[0], ALL_Pos_QPL[1], ALL_Pos_QPL[2], ALL_Pos_QPL[3], ALL_Pos_QPL[4],
             ALL_Pos_QPL[5], ALL_Pos_QPL[6], ALL_Pos_QPL[7], ALL_Pos_QPL[8], ALL_Pos_QPL[9],
             ALL_Pos_QPL[10], ALL_Pos_QPL[11], ALL_Pos_QPL[12], ALL_Pos_QPL[13], ALL_Pos_QPL[14],
             ALL_Pos_QPL[15], ALL_Pos_QPL[16], ALL_Pos_QPL[17], ALL_Pos_QPL[18], ALL_Pos_QPL[19], ALL_Pos_QPL[20],
             ALL_Neg_QPL[0], ALL_Neg_QPL[1], ALL_Neg_QPL[2], ALL_Neg_QPL[3], ALL_Neg_QPL[4],
             ALL_Neg_QPL[5], ALL_Neg_QPL[6], ALL_Neg_QPL[7], ALL_Neg_QPL[8], ALL_Neg_QPL[9],
             ALL_Neg_QPL[10], ALL_Neg_QPL[11], ALL_Neg_QPL[12], ALL_Neg_QPL[13], ALL_Neg_QPL[14],
             ALL_Neg_QPL[15], ALL_Neg_QPL[16], ALL_Neg_QPL[17], ALL_Neg_QPL[18], ALL_Neg_QPL[19], ALL_Neg_QPL[20]);
         // Store the training data into the file.
         //FileWrite(Currency_QPL_FileHandle, IntegerToString(DT_YY[d - 1]) + "/" + IntegerToString(DT_MM[d - 1]) + "/" + IntegerToString(DT_DD[d - 1]),
         //    DoubleToString(DT_OP[d - 1], TP_nD[nTP]), DoubleToString(DT_HI[d - 1], TP_nD[nTP]), DoubleToString(DT_LO[d - 1], TP_nD[nTP]), DoubleToString(DT_CL[d - 1], TP_nD[nTP]), DoubleToString(DT_VL[d-1], TP_nD[nTP]));
      }
      // Close file.
      FileClose(Currency_QPL_FileHandle);
   }
   
   // Check Global Time.
   Getime = GetTickCount();
   Gtlapse = Getime - Gstime;
   
   // Output time taken.
   Print("Total Time Taken : ", Gtlapse, " msec");
   
   // Return the result for initialized successful.
   return(INIT_SUCCEEDED);
}

// Main Function.
void OnTick(){}