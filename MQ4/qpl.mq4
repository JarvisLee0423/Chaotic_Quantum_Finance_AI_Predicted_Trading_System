//****************************************************************************************************************************
// Date        : 6 Jan 2019                                                                                               
// Created by  : Dr. Raymond LEE                                                                                             
// Name        : QPL Cacluation Program        
//               VERSION NO: 1.1            
// Objective   : This program calculate the N(QPR) of ALL 120 Financial Products
//               For Each Financial Product:
//               1) Read the Daily Time Series and extract (Date, O, H, L, C, V) m
//               2) Calculate Dally Price Return r(t)
//               3) Calculate quantum price return wavefunction Q(r)(size 100)
//               4) Evalutate lambda (L) value for the wavefunction Q(r)using F.D.M. at ground state
//                  L = abs((r0^2*Q0 - r1^2*Q1)/(r0^4*Q0 - r1^4*Q1))
//               5) Evaluate other related parameters:  
//                  - sigma  (std dev of Q)
//                  - maxQPR (max Quantum Price Return - for normalization)       
//               6) Once L is found, using Quartic Schrodinger Equation of Quantum Finance to find 
//                  all the 21 Quantum Price Energies (QFEL0 .. QFEL20). 
//                  Given by:
//                  (E(n)/(2n+1))^3 - (E(n)/(2n+1)) - K(n)^3 * L = 0
//                  where 
//                   K(n) = ((1.1924+33.2383n+56.2169n^2)/(1+43.6106n))^(1/3)
//               7) Solve the 21 Cubic Eqts in (6) and extract the +ve real roots as QFEL0 .. QFEL20.
//               8) Cacluate QPR(n)  = QFEL(n)/QFEL(0) n = [1 .. 20]
//               9) Cacluate NQPR(n) = 1 + 0.21*sigma*QPR(n);
//               10)Save TWO Level of datafiles
//                  1) For each financial product, save the QPL Table contains
//                     QFEL, QPR, NQPR for the first 21 energy levels
//                  2) For all financial product, create a QPL Summary table contains NQPR for all FP                                                      
//                     
//                                                                                                                           
//****************************************************************************************************************************

#property copyright "Copyright © 2019, DR. RAYMOND LEE"
#property link      "http://QFFC.ORG"

// DEFINE DIRECTORIES
string      QP_Directory   = "QPL";       // QPL Directory
string      QPData_Directory  = "QPL_Data";  // QPL Directory
string      TS_FileName = "";             // File name for TimeSeries
int         TS_FileHandle;                // File Handle for TimeSeries
string      Qf_FileName = "";             // File name for Qfunction
int         Qf_FileHandle;                // File Handle for Qfunction
string      ALL_QFEL_FileName = "";        // File name for ALL_QFEL
int         ALL_QFEL_FileHandle;           // File Handle for ALL_QFEL
string      ALL_QPR_FileName = "";        // File name for ALL_QPR
int         ALL_QPR_FileHandle;           // File Handle for ALL_QPR
string      ALL_NQPR_FileName = "";       // File name for ALL_NQPR
int         ALL_NQPR_FileHandle;          // File Handle for ALL_NQPR
string      QPD_FileName = "";            // File name for QPL Details
int         QPD_FileHandle;               // File Handle for QPL Details
string      QPLog_FileName = "";          // File name for QPL Details
int         QPLog_FileHandle;             // File Handle for QPL Details
string      Lambda_FileName = "";          // File name for QPL Details
int         Lambda_FileHandle;             // File Handle for QPL Details

string      Currency_QPL_FileName = "";   //File name for each currency
int         Currency_QPL_FileHandle;      //File Handl for each currency

// DEFINE GLOBAL VARIABLES
int         maxELevel  = 21;              // Max Energy Level      
int         maxTP      = 10;             // Max no of Financial Product
int         maxTS      = 2048;            // Max no of Time Series Record
int         nTP=0;
double      p3=1.0/3.0;                       // Set 1/3 for MathPow

//DEFINE TIMING VARIABLES
uint        stime=0;
uint        etime=0;
uint        Gstime=0;
uint        Getime=0;
uint        tlapse=0;
uint        Gtlapse=0;

// DEFINE FINANCIAL PRODUCT RELATED VARIABLES
string      TPSymbol   = "";              // Current Trading duct Symbol

//string      TP_Code[120]={"XAGUSD","CORN","US30","AUDUSD","EURCHF","GBPCAD","NZDJPY","USDCNH","XAUAUD","XAUCHF",
//                          "XAUEUR","XAUGBP","XAUJPY","XAUUSD","COPPER","PALLAD","PLAT","UK_OIL","US_OIL","US_NATG",
//                          "HTG_OIL","COTTON","SOYBEAN","SUGAR","WHEAT","IT40","AUS200","CHINAA50","ESP35","ESTX50",
//                          "FRA40","GER30","HK50","JPN225","N25","NAS100","SIGI","SPX500","SWISS20","UK100","US2000",
//                          "AUDCAD","AUDCHF","AUDCNH","AUDJPY","AUDNOK","AUDNZD","AUDPLN","AUDSGD","CADCHF","CADJPY",
//                          "CADNOK","CADPLN","CHFHUF","CHFJPY","CHFNOK","CHFPLN","CNHJPY","EURAUD","EURCAD","EURCNH",
//                          "EURCZK","EURDKK","EURGBP","EURHKD","EURHUF","EURJPY","EURMXN","EURNOK","EURNZD","EURPLN",
//                          "EURRON","EURRUB","EURSEK","EURSGD","EURTRY","EURUSD","EURZAR","GBPAUD","GBPCHF","GBPDKK",
//                          "GBPHKD","GBPJPY","GBPMXN","GBPNOK","GBPNZD","GBPPLN","GBPSEK","GBPSGD","GBPUSD","GBPZAR",
//                          "HKDJPY","NOKDKK","NOKJPY","NOKSEK","NZDCAD","NZDCHF","NZDUSD","SGDHKD","SGDJPY","TRYJPY",
//                          "USDCAD","USDCHF","USDCZK","USDDKK","USDHKD","USDHUF","USDILS","USDJPY","USDMXN","USDNOK",
//                          "USDPLN","USDRON","USDRUB","USDSEK","USDSGD","USDTHB","USDTRY","USDZAR","ZARJPY"};
                                               
//int         TP_No[120]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
//                        31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
//                        61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,
//                        91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,
//                        116,117,118,119,120};
//                        
//int         TP_nD[120]={3,2,1,5,5,5,3,5,2,2,2,2,0,2,2,2,2,0,2,3,0,2,2,2,2,0,0,0,0,0,1,1,0,0,2,1,1,1,0,1,1,5,5,4,
//                        3,4,5,5,5,5,3,5,5,3,3,5,5,4,5,5,4,4,5,5,4,3,3,5,5,5,5,4,3,5,4,5,5,5,5,5,5,4,3,4,5,5,5,5,
//                        5,5,4,4,5,4,5,5,5,5,4,3,3,5,5,3,5,5,3,5,3,5,5,5,5,3,5,5,3,5,5,3};
//---

string      TP_Code[10]={"AUDJPY","CADCHF","EURAUD","EURCAD","EURUSD","GBPUSD","USDCAD","CADJPY","USDHKD","USDJPY"};
int         TP_No[10]={1,2,3,4,5,6,7,8,9,10};
int         TP_nD[10]={3,5,5,5,5,5,5,3,5,3};

// Deinit
int deinit() {
   return(0);
}

int init() {
   int      eL, d, TSsize, nQ, maxQno, nR, maxRno, tQno;
   double   auxR, maxQ, r0, r1, rn1;
   double   dr, Lup, Ldw, L, mu, sigma; 
   bool     bFound;
   
   // Variables used in Cardano's Method
   double   p, q, u, v;
         
   // Declare Time Series Array
   int        DT_YY[2048];
   int        DT_MM[2048];
   int        DT_DD[2048];
   double     DT_OP[2048];
   double     DT_HI[2048];
   double     DT_LO[2048];
   double     DT_CL[2048];
   double     DT_VL[2048];
   double     DT_RT[2048];
      
   // Declare Array for Quantum Price Wavefunction 
   double     Q[100];            // Quantum Price Wavefunction
   double     NQ[100];           // Normalized Q[]
   double     r[100];            // r no
   
   // Declare ARRAY for QPL related arrays
   double     ALL_QFEL[10][21];  // Array contains QFEL  for all FPs
   double     ALL_QPR[10][21];   // Array contains QPR  for all FPs
   double     ALL_NQPR[10][21];  // Array contains NQPR for all FPs
   double     QFEL[21];           // QFEL for each FP
   double     QPR[21];            // QPR  for each FP
   double     NQPR[21];           // NQPR for each FP
   double     K[21];              // K values in QP Schrodinger Eqt
   
   double     ALL_Pos_QPL[21];
   double     ALL_Neg_QPL[21];
    
   // Set Global Start Time
   Gstime       = GetTickCount();

   // CREATE QPL SUMMARY DATA FILE
   ALL_QFEL_FileName      = "FX_QFEL.csv";   
   FileDelete(QP_Directory+"//"+ALL_QFEL_FileName,FILE_COMMON);
   ResetLastError();
   ALL_QFEL_FileHandle    = FileOpen(QP_Directory+"//"+ALL_QFEL_FileName, FILE_COMMON|FILE_READ|FILE_WRITE|FILE_CSV,',');

   // CREATE QPL SUMMARY DATA FILE
   ALL_QPR_FileName      = "FX_QPR.csv";   
   FileDelete(QP_Directory+"//"+ALL_QPR_FileName,FILE_COMMON);
   ResetLastError();
   ALL_QPR_FileHandle    = FileOpen(QP_Directory+"//"+ALL_QPR_FileName, FILE_COMMON|FILE_READ|FILE_WRITE|FILE_CSV,',');

   // CREATE QPL SUMMARY DATA FILE
   ALL_NQPR_FileName      = "FX_NQPR.csv";   
   FileDelete(QP_Directory+"//"+ALL_NQPR_FileName,FILE_COMMON);
   ResetLastError();
   ALL_NQPR_FileHandle   = FileOpen(QP_Directory+"//"+ALL_NQPR_FileName,FILE_COMMON|FILE_READ|FILE_WRITE|FILE_CSV,',');
   
     
   // Write Header Line for ALL_QFEL DataFile
   FileWrite(ALL_QFEL_FileHandle,"CODE","QFEL[0]",
             "QFEL[1]", "QFEL[2]", "QFEL[3]", "QFEL[4]", "QFEL[5]",
             "QFEL[6]", "QFEL[7]", "QFEL[8]", "QFEL[9]", "QFEL[10]",
             "QFEL[11]","QFEL[12]","QFEL[13]","QFEL[14]","QFEL[15]",
             "QFEL[16]","QFEL[17]","QFEL[18]","QFEL[19]","QFEL[20]");

   // Write Header Line for ALL_QPR DataFile
   FileWrite(ALL_QPR_FileHandle,"CODE","QPR[0]",
             "QPR[1]", "QPR[2]", "QPR[3]", "QPR[4]", "QPR[5]",
             "QPR[6]", "QPR[7]", "QPR[8]", "QPR[9]", "QPR[10]",
             "QPR[11]","QPR[12]","QPR[13]","QPR[14]","QPR[15]",
             "QPR[16]","QPR[17]","QPR[18]","QPR[19]","QPR[20]");
                                  
   // Write Header Line for ALL_NQPR DataFile
   // FileWrite(ALL_NQPR_FileHandle,"CODE","NQPR[0]",
   //          "NQPR[1]", "NQPR[2]", "NQPR[3]", "NQPR[4]", "NQPR[5]",
   //          "NQPR[6]", "NQPR[7]", "NQPR[8]", "NQPR[9]", "NQPR[10]",
   //          "NQPR[11]","NQPR[12]","NQPR[13]","NQPR[14]","NQPR[15]",
   //          "NQPR[16]","NQPR[17]","NQPR[18]","NQPR[19]","NQPR[20]");
                 
   // CREATE QP Log Filel
   QPLog_FileName    = "QPL_120Log.csv";   
   FileDelete(QP_Directory+"//"+QPLog_FileName,FILE_COMMON);
   ResetLastError();
   QPLog_FileHandle  = FileOpen(QP_Directory+"//"+QPLog_FileName,FILE_COMMON|FILE_READ|FILE_WRITE|FILE_CSV,',');

   // CREATE Lambda FILE
   Lambda_FileName      = "Lambda_FX.csv";   
   FileDelete(QP_Directory+"//"+Lambda_FileName,FILE_COMMON);
   ResetLastError();
   Lambda_FileHandle   = FileOpen(QP_Directory+"//"+Lambda_FileName,FILE_COMMON|FILE_READ|FILE_WRITE|FILE_CSV,',');

   // Write Header Line for Lambda File
   FileWrite(Lambda_FileHandle,"CODE","Lambda");

   //*******************************************************************
   //
   // 1. Cacluate All K values [K0 .. K20] using the following formula:
   //
   //  K[eL] = pow((1.1924 + 33.2383*eL + 56.2169*eL*eL)/(1 + 43.6106 *eL),p3);
   //
   
   // Printout K List Header
   Print("Printout ALL K values K0 .. K20 for first 20 Energy Levels");
   FileWrite(QPLog_FileHandle,"Printout ALL K values K0 .. K20 for first 20 Energy Levels");
      
   for (eL=0;eL<21;eL++)
   {
      K[eL] = MathPow((1.1924 + (33.2383*eL) + (56.2169*eL*eL))/(1 + (43.6106 *eL)),p3);
      Print("Energy Level ",eL," K",eL," = ",K[eL]);
      FileWrite(QPLog_FileHandle,"Energy Level ",eL," K",eL," = ",K[eL]);   
   }
      
   // ****************************************
   // LOOP OVER ALL TP
   for (nTP=0;nTP<maxTP;nTP++)
   {
      TPSymbol    = TP_Code[nTP];         // Get TP Symbol
      stime       = GetTickCount();       // Get timer

      // CREATE QPL Detail DATA FILE
      QPD_FileName      = TP_No[nTP]+" "+TPSymbol+"_QPR.csv";  
      FileDelete(QP_Directory+"//"+QPD_FileName,FILE_COMMON);
      ResetLastError();
      QPD_FileHandle    = FileOpen(QPData_Directory+"//"+QPD_FileName,FILE_COMMON|FILE_READ|FILE_WRITE|FILE_CSV,',');

      // Write Header Line
      FileWrite(QPD_FileHandle,"Year","Month","Day","Open","High","Low","Close","Volumn","Return");
      
      // CREATE Qf Wavefunction Distribution DataFile
      Qf_FileName      = TP_No[nTP]+" "+TPSymbol+"_Qf.csv";  
      FileDelete(QP_Directory+"//"+Qf_FileName,FILE_COMMON);
      ResetLastError();
      Qf_FileHandle   = FileOpen(QP_Directory+"//"+Qf_FileName,FILE_COMMON|FILE_READ|FILE_WRITE|FILE_CSV,',');

      // Write Header Line of Qfile
      FileWrite(Qf_FileHandle,"r","Q(r)","NQ(r)");
      
      Currency_QPL_FileName      = TP_Code[nTP]+"_Data.csv";   
      FileDelete(QP_Directory+"//New//"+Currency_QPL_FileName,FILE_COMMON);
      ResetLastError();
      Currency_QPL_FileHandle   = FileOpen(QP_Directory+"//New//"+Currency_QPL_FileName,FILE_COMMON|FILE_READ|FILE_WRITE|FILE_CSV,',');
          
      //********************************************************************************************************
      //
      // 2. READ ALL Daily Time Series 
      //
      //********************************************************************************************************
      
      // Since iBars/Bars doesn't work, manually check TSsize
      TSsize = 0;
      while (iTime(TPSymbol,PERIOD_D1,TSsize)>0 && (TSsize<maxTS))
      {
         TSsize++;
      }  
      
      // Using For LOOP to get all the time series data
      for (d=1;d<TSsize;d++)
      {
          DT_YY[d-1] = TimeYear(iTime(TPSymbol,PERIOD_D1,d));     
          DT_MM[d-1] = TimeMonth(iTime(TPSymbol,PERIOD_D1,d));     
          DT_DD[d-1] = TimeDay(iTime(TPSymbol,PERIOD_D1,d));     
          DT_OP[d-1] = iOpen(TPSymbol,PERIOD_D1,d);
          DT_HI[d-1] = iHigh(TPSymbol,PERIOD_D1,d);
          DT_LO[d-1] = iLow(TPSymbol,PERIOD_D1,d);
          DT_CL[d-1] = iClose(TPSymbol,PERIOD_D1,d);
          DT_VL[d-1] = iVolume(TPSymbol,PERIOD_D1,d);
          DT_RT[d-1] = 1;
      }
      
      // Cacluate DT_RT[d]
      for (d=0;d<(TSsize-2);d++)
      {
          if (DT_CL[d+1] > 0) 
          {
            DT_RT[d] = DT_CL[d]/DT_CL[d+1];
          }else{
            DT_RT[d] = 1;
          }
          
          // Write out the QPD data file
          FileWrite(QPD_FileHandle,DT_YY[d],DT_MM[d],DT_DD[d],
             DoubleToString(DT_OP[d],TP_nD[nTP]),DoubleToString(DT_HI[d],TP_nD[nTP]),DoubleToString(DT_LO[d],TP_nD[nTP]),DoubleToString(DT_CL[d],TP_nD[nTP]),
             DT_VL[d], DoubleToString(DT_RT[d],8));
      }
      
      // Close QP Detail Data File
      FileClose(QPD_FileHandle);
      
      //******************************************************************
      //
      // 3. Calculate Mean (mu) and Standard Deviation (sigma) of return array
      //
      //*******************************************************************
      
      maxRno = TSsize - 2;
      
      // Calculate mean mu first
      mu = 0;
      for (d=0;d<maxRno;d++)
      {
        mu = mu + DT_RT[d];
      }
      mu = mu/maxRno;
      
      // Calculate STDEV sigma
      sigma = 0;
      for (d=0;d<maxRno;d++)
      {
        sigma = sigma + (DT_RT[d]-mu)*(DT_RT[d]-mu);
      }
      sigma = sqrt((sigma / maxRno));      
      
      // Calculate dr where dr = 3*sigma/50
      dr = 3 * sigma / 50;
      
      Print("TP",nTP+1," ",TP_Code[nTP]," No of r = ",maxRno," mu = ",mu," sigma = ",sigma," dr=",dr);
      FileWrite(QPLog_FileHandle, "TP",nTP+1," ",TP_Code[nTP]," No of r = ",maxRno," mu = ",mu," sigma = ",sigma," dr=",dr);
      
      //******************************************************************
      //
      // 4. Generate the QP Wavefunction distribution 
      //
      //*******************************************************************
      auxR   = 0;
      
      // Loop over all r from r - 50*dr to r + 50*dr and get the distribution function
      // Reset all the Q[] first
      for (nQ=0;nQ<100;nQ++)
      {
        Q[nQ] = 0;
      }
      
      // Loop over the maxRno to get the distribution
      tQno = 0;
      for (nR=0;nR<maxRno;nR++)
      {
        bFound = False;
        nQ = 0;
        auxR = 1 - (dr * 50);
        while (!bFound && (nQ < 100))
        {
           if ((DT_RT[nR] > auxR) && (DT_RT[nR] <= (auxR + dr)))
           {
              Q[nQ]++;
              tQno++;
              bFound = True;
           }else
           {
              nQ++;
              auxR = auxR + dr;
           }
        }
      }
      
      // Write out the Qfile for Record
      auxR = 1 - (dr * 50);
      for (nQ=0;nQ<100;nQ++)
      {
         r[nQ]  = auxR;
         NQ[nQ] = Q[nQ]/tQno;        
         FileWrite(Qf_FileHandle,auxR,Q[nQ],NQ[nQ]);        
         auxR = auxR + dr;
      }       
      
      // Find maxQ and maxQno
      maxQ   = 0;
      maxQno = 0;
      for (nQ=0;nQ<100;nQ++)
      {
         if (NQ[nQ] > maxQ)
         {
            maxQ   = NQ[nQ];
            maxQno = nQ; 
         }       
      }    
      
      // Printout the maxQ, maxQno       
      Print("TP",nTP+1," ",TP_Code[nTP]," MaxQ= ",maxQ," maxQno=",maxQno," Total Qno =",tQno);
      FileWrite(QPLog_FileHandle,"TP",nTP+1," ",TP_Code[nTP]," MaxQ= ",maxQ," maxQno=",maxQno," Total Qno =",tQno);

      //******************************************************************
      //
      // 5. Evaluate Lambda L for the QP Wavefuntion
      //
      //*******************************************************************
      //     
      // Given maxQno - i.e. ground state Q[0], r[0] = r[maxQno-dr]
      // We have Q[+1] = NQ[maxQno+1], r[+1] = r[maxQno]+(dr/2)
      //         Q[-1] = NQ[maxQno-1], r[-1] = r[maxQno]-(dr*1.5)
      // Apply F.D.M. into QP Sch Eqtuation
      // L = abs((r[-1]^2*Q[-1]-(r[+1]^2*Q[+1]))/(r[-1]^4*Q[-1]-(r[+1]^4*Q[+1])))
      
      r0  = r[maxQno] - (dr/2);
      r1  = r0 + dr;
      rn1 = r0 - dr;
      Lup = (pow(rn1,2)*NQ[maxQno-1])-(pow(r1,2)*NQ[maxQno+1]);
      Ldw = (pow(rn1,4)*NQ[maxQno-1])-(pow(r1,4)*NQ[maxQno+1]);
      L   = MathAbs(Lup/Ldw);

      // Printout r0,Q0, r1, Q1, r-1 Q-1
      Print("TP",nTP+1," ",TP_Code[nTP]," r0 = ",r0," r1 = ",r1," r-1 = ",rn1," Q0 = ",NQ[maxQno]," Q1 = ",NQ[maxQno+1]," Q-1 = ",NQ[maxQno-1]," L = Lup/Ldw = ",Lup,"/",Ldw," = ",L);
      FileWrite(QPLog_FileHandle," r0 = ",r0," r1 = ",r1," r-1 = ",rn1," Q0 = ",NQ[maxQno]," Q1 = ",NQ[maxQno+1]," Q-1 = ",NQ[maxQno-1]," L = Lup/Ldw = ",Lup,"/",Ldw," = ",L);
      FileWrite(Lambda_FileHandle,TPSymbol,L);
      Print(TPSymbol,L);

      //******************************************************************
      //
      // 6. Using QP Schrodinger Eqt to FIND first 21 Energy Levels
      //
      //    By solving the Quartic Anharmonic Oscillator as cubic polynomial eqt
      //    of the form
      //
      //        a*x^3 + b*x^2 + c*x + d = 0
      //
      //    Using (Dasqupta et. al. 2007) QAHO solving equation:
      // 
      //    (E(n)/(2n+1))^3 - (E(n)/(2n+1)) - K(n)^3*L = 0
      //
      //    Solving the above Depressed Cubic Eqt using Cardano's Method
      //    
      //    Given    t^3 + p*t + q = 0
      //    Let      t = u + v
      //    Cardano's Method deduced that:
      //        u^3 = -q/2 + sqrt(q^2/4 + p^3/27)
      //        v^3 = -q/2 - sqrt(q^2/4 + p^3/27)
      //    The first cubic root (real root) will be:
      //
      //        t = u + v        
      //
      //    So, combining Cardano's Method into our QF Sch Eqt. 
      //    We have
      //    Substitue p = -(2n+1)^2;  q = -L(2n+1)^3*(K(n)^3) into the above equations to get the 
      //    real root
      //
      //*********************************************************************************************
      
      for (eL=0;eL<21;eL++)
      {
         p = -1 * pow((2*eL+1),2);
         q = -1 * L * pow((2*eL+1),3) * pow(K[eL],3);
         
         // Apply Cardano's Method to find the real root of the depressed cubic equation
         u = MathPow((-0.5*q + MathSqrt(((q*q/4.0) + (p*p*p/27.0)))),p3);
         v = MathPow((-0.5*q - MathSqrt(((q*q/4.0) + (p*p*p/27.0)))),p3);
         
         // Store the QFEL 
         QFEL[eL] = u + v;
         
         // Printout the QF Energy Levels
         Print("TP",nTP+1," ",TP_Code[nTP]," Energy Level",eL," QFEL = ",QFEL[eL]);
         FileWrite(QPLog_FileHandle," Energy Level",eL," QFEL = ",QFEL[eL]);
      }
      
      // Evaluate ALL QPR values
      for (eL=0;eL<21;eL++)
      {     
         QPR[eL]  = QFEL[eL]/QFEL[0];
         NQPR[eL] = 1 + 0.21*sigma*QPR[eL];
         
         // Store into ALL QFEL, QPR, NQPR into array 
         ALL_QFEL[nTP,eL] = QFEL[eL]; 
         ALL_QPR[nTP,eL]  = QPR[eL]; 
         ALL_NQPR[nTP,eL] = NQPR[eL]; 
         
      }
      
      FileWrite(Currency_QPL_FileHandle,"Date", "Open", "High", "Low", "Close","QPL_1","QPL_2","QPL_3","QPL_4","QPL_5","QPL_6","QPL_7","QPL_8",
      "QPL_9","QPL_10","QPL_11","QPL_12","QPL_13","QPL_14","QPL_15","QPL_16","QPL_17","QPL_18","QPL_19","QPL_20","QPL_21","QPL_-1","QPL_-2",
      "QPL_-3","QPL_-4","QPL_-5","QPL_-6","QPL_-7","QPL_-8","QPL_-9","QPL_-10","QPL_-11","QPL_-12","QPL_-13","QPL_-14","QPL_-15","QPL_-16",
      "QPL_-17","QPL_-18","QPL_-19","QPL_-20","QPL_-21");
      // Using For LOOP to get all the time series data
      for (d=1;d<TSsize;d++)
      {
         for (eL=0;eL<21;eL++)
         {     
            ALL_Pos_QPL[eL] = DT_OP[d-1] * NQPR[eL];
            ALL_Neg_QPL[eL] = DT_OP[d-1] / NQPR[eL];
         }     
         FileWrite(Currency_QPL_FileHandle,IntegerToString(DT_YY[d-1])+"/"+IntegerToString(DT_MM[d-1])+"/"+IntegerToString(DT_DD[d-1]),
             DoubleToString(DT_OP[d-1],TP_nD[nTP]),DoubleToString(DT_HI[d-1],TP_nD[nTP]),DoubleToString(DT_LO[d-1],TP_nD[nTP]),DoubleToString(DT_CL[d-1],TP_nD[nTP]),
             ALL_Pos_QPL[0],ALL_Pos_QPL[1],ALL_Pos_QPL[2],ALL_Pos_QPL[3],ALL_Pos_QPL[4],
             ALL_Pos_QPL[5],ALL_Pos_QPL[6],ALL_Pos_QPL[7],ALL_Pos_QPL[8],ALL_Pos_QPL[9],
             ALL_Pos_QPL[10],ALL_Pos_QPL[11],ALL_Pos_QPL[12],ALL_Pos_QPL[13],ALL_Pos_QPL[14],
             ALL_Pos_QPL[15],ALL_Pos_QPL[16],ALL_Pos_QPL[17],ALL_Pos_QPL[18],ALL_Pos_QPL[19],ALL_Pos_QPL[20],
             ALL_Neg_QPL[0],ALL_Neg_QPL[1],ALL_Neg_QPL[2],ALL_Neg_QPL[3],ALL_Neg_QPL[4],
             ALL_Neg_QPL[5],ALL_Neg_QPL[6],ALL_Neg_QPL[7],ALL_Neg_QPL[8],ALL_Neg_QPL[9],
             ALL_Neg_QPL[10],ALL_Neg_QPL[11],ALL_Neg_QPL[12],ALL_Neg_QPL[13],ALL_Neg_QPL[14],
             ALL_Neg_QPL[15],ALL_Neg_QPL[16],ALL_Neg_QPL[17],ALL_Neg_QPL[18],ALL_Neg_QPL[19],ALL_Neg_QPL[20]);         
      }
      
      
      // Close Qfile
      FileClose(Qf_FileHandle);   

      // Write out ALL_QFEL into QFEL Datafile
      FileWrite(ALL_QFEL_FileHandle,TPSymbol,ALL_QFEL[nTP,0],
                ALL_QFEL[nTP,1], ALL_QFEL[nTP,2], ALL_QFEL[nTP,3], ALL_QFEL[nTP,4], ALL_QFEL[nTP,5],
                ALL_QFEL[nTP,6], ALL_QFEL[nTP,7], ALL_QFEL[nTP,8], ALL_QFEL[nTP,9], ALL_QFEL[nTP,10],
                ALL_QFEL[nTP,11],ALL_QFEL[nTP,12],ALL_QFEL[nTP,13],ALL_QFEL[nTP,14],ALL_QFEL[nTP,15],
                ALL_QFEL[nTP,16],ALL_QFEL[nTP,17],ALL_QFEL[nTP,18],ALL_QFEL[nTP,19],ALL_QFEL[nTP,20]);     
 
      // Write out ALL_QPR into QPR Datafile
      FileWrite(ALL_QPR_FileHandle,TPSymbol,ALL_QPR[nTP,0],
                ALL_QPR[nTP,1], ALL_QPR[nTP,2], ALL_QPR[nTP,3], ALL_QPR[nTP,4], ALL_QPR[nTP,5],
                ALL_QPR[nTP,6], ALL_QPR[nTP,7], ALL_QPR[nTP,8], ALL_QPR[nTP,9], ALL_QPR[nTP,10],
                ALL_QPR[nTP,11],ALL_QPR[nTP,12],ALL_QPR[nTP,13],ALL_QPR[nTP,14],ALL_QPR[nTP,15],
                ALL_QPR[nTP,16],ALL_QPR[nTP,17],ALL_QPR[nTP,18],ALL_QPR[nTP,19],ALL_QPR[nTP,20]);     
      
      // Write out ALL_NQPR into NQPR Datafile
      FileWrite(ALL_NQPR_FileHandle,ALL_NQPR[nTP,0],
                ALL_NQPR[nTP,1], ALL_NQPR[nTP,2], ALL_NQPR[nTP,3], ALL_NQPR[nTP,4], ALL_NQPR[nTP,5],
                ALL_NQPR[nTP,6], ALL_NQPR[nTP,7], ALL_NQPR[nTP,8], ALL_NQPR[nTP,9], ALL_NQPR[nTP,10],
                ALL_NQPR[nTP,11],ALL_NQPR[nTP,12],ALL_NQPR[nTP,13],ALL_NQPR[nTP,14],ALL_NQPR[nTP,15],
                ALL_NQPR[nTP,16],ALL_NQPR[nTP,17],ALL_NQPR[nTP,18],ALL_NQPR[nTP,19],ALL_NQPR[nTP,20]);   
      FileClose(Currency_QPL_FileHandle);  
   } // Main Loop for FP
   
   // Close All DataFiles
   FileClose(QPLog_FileHandle);
   FileClose(Lambda_FileHandle);
   FileClose(ALL_NQPR_FileHandle);
   
   // Check Global Time
   Getime   = GetTickCount();
   Gtlapse  = Getime - Gstime;
   
   // Output time taken
   Print("Total Time Taken : ",Gtlapse," msec");
   
   return(0);
}

int start()
  {   

   return(0);
  }

