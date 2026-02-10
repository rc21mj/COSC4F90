//
//                  Simu5G
//
// Authors: Giovanni Nardini, Giovanni Stea, Antonio Virdis (University of Pisa)
//
// This file is part of a software released under the license included in file
// "license.pdf". Please read LICENSE and README files before using it.
// The above files and the present reference are part of the software itself,
// and cannot be removed from it.
//

#ifndef _LTE_AIRPHYUE_H_
#define _LTE_AIRPHYUE_H_

#include "stack/phy/layer/LtePhyBase.h"
#include "stack/phy/das/DasFilter.h"
#include "stack/mac/layer/LteMacUe.h"
#include "stack/rlc/um/LteRlcUm.h"
#include "stack/pdcp_rrc/layer/LtePdcpRrc.h"

#include "stack/phy/ChannelModel/LteRealisticChannelModel.h"
#include "stack/phy/layer/NazaninHandoverDecision.h"
#include <inet/common/INETDefs.h>
#include <vector>
#include <numeric>
#include <map>



class DasFilter;

class LtePhyUe : public LtePhyBase
{
  protected:
    LtePhyUe* otherPhy_;

    /** Master MacNodeId */
    MacNodeId masterId_;

    /** Statistic for serving cell */
    omnetpp::simsignal_t servingCell_;

    /** Self message to trigger handover procedure evaluation */
    omnetpp::cMessage *handoverStarter_;

    /** Self message to start the handover procedure */
    omnetpp::cMessage *handoverTrigger_;

    /** RSSI received from the current serving node */
    double currentMasterRssi_;
    ///////////////////////
    double currentMasterSinr_;
    double currentMasterRsrp_;
    double currentMasterDist_;
    double currentMasterSpeed_;
    ///////////////////////

    /** ID of not-master node from which highest RSSI was received */
    MacNodeId candidateMasterId_;

    /** Highest RSSI received from not-master node */
    double candidateMasterRssi_;
    double candidateMasterSinr_;
    double candidateMasterRsrp_;
    double candidateMasterDist_;
    double candidateMasterSpeed_;
    double hysteresislaodTh_;

    /**
     * Hysteresis threshold to evaluate handover: it introduces a small polarization to
     * avoid multiple subsequent handovers
     */
    double hysteresisTh_; //HO threshold value
    double hysteresisSinrTh_; //HO threshold value
    double hysteresisRsrpTh_; //HO threshold value
    double hysteresisDistTh_; //HO threshold value
    double hysteresisSpeedTh_; //HO threshold value
    double hysteresisLoadTh_; //HO threshold value

    /**
     * Value used to divide currentMasterRssi_ and create an hysteresisTh_
     * Use zero to have hysteresisTh_ == 0.
     */
    // TODO: bring it to ned par!
    double hysteresisFactor_;
    double hysteresisFactorSinr_;
    double hysteresisFactorRsrp_;
    double hysteresisFactorDist_;
    double hysteresisFactorLoad_;

    /**
     * Time interval elapsing from the reception of first handover broadcast message
     * to the beginning of handover procedure.
     * It must be a small number greater than 0 to ensure that all broadcast messages
     * are received before evaluating handover.
     * Note that broadcast messages for handover are always received at the very same time
     * (at bdcUpdateInterval_ seconds intervals).
     */
    // TODO: bring it to ned par!
    double handoverDelta_;

    // time for completion of the handover procedure
    double handoverLatency_;
    double handoverDetachment_;
    double handoverAttachment_;
    double hoVC = 1.7;

    // lower threshold of RSSI for detachment
    double minRssi_;
    double minSinr_;
    double minRsrp_;
    double maxDist_;

    //count HO for each vehicle
    int countVehicleHO = 0;

//    int speed0to16 = 0, speed16to33 = 0, speed33tohigh = 0;
    /**
     * Handover switch
     */
    bool enableHandover_;

    /**
     * Pointer to the DAS Filter: used to call das function
     * when receiving broadcasts and to retrieve physical
     * antenna properties on packet reception
     */
    DasFilter* das_;

    /// Threshold for antenna association
    // TODO: bring it to ned par!
    double dasRssiThreshold_;

    /** set to false if a battery is not present in module or must have infinite capacity */
    bool useBattery_;
    double txAmount_;    // drawn current amount for tx operations (mA)
    double rxAmount_;    // drawn current amount for rx operations (mA)

    LteMacUe *mac_;
    LteRlcUm *rlcUm_;
    LtePdcpRrcBase *pdcp_;

    omnetpp::simtime_t lastFeedback_;

    // support to print averageCqi at the end of the simulation
    std::vector<short int> cqiDlSamples_;
    std::vector<short int> cqiUlSamples_;
    unsigned int cqiDlSum_;
    unsigned int cqiUlSum_;
    unsigned int cqiDlCount_;
    unsigned int cqiUlCount_;

    virtual void initialize(int stage) override;
    virtual void handleSelfMessage(omnetpp::cMessage *msg) override;
    virtual void handleAirFrame(omnetpp::cMessage* msg) override;
    virtual void finish() override;
    virtual void finish(cComponent *component, omnetpp::simsignal_t signalID) override {cIListener::finish(component, signalID);}

    virtual void handleUpperMessage(omnetpp::cMessage* msg) override;


    /**
     * Utility function to update the hysteresis threshold using hysteresisFactor_.
     */
//    double updateHysteresisTh(double v);
//    double updateHysteresisThMinSinr(double v);
//    double updateHysteresisThMinRsrp(double v);
//    double updateHysteresisThMaxDist(double v);
//    double updateHysteresisTowerLoad(double v);
    void handoverHandler(LteAirFrame* frame, UserControlInfo* lteInfo);

    void deleteOldBuffers(MacNodeId masterId);

    virtual void triggerHandover();
    virtual void doHandover();

  public:

    LtePhyUe();
        virtual ~LtePhyUe();

        DasFilter *getDasFilter();
        /**
         * Send Feedback, called by feedback generator in DL
         */
        virtual void sendFeedback(LteFeedbackDoubleVector fbDl, LteFeedbackDoubleVector fbUl, FeedbackRequest req);
        MacNodeId getMasterId() const
        {
            return masterId_;
        }
        omnetpp::simtime_t coherenceTime(double speed)
        {
            double fd = (speed / SPEED_OF_LIGHT) * carrierFrequency_;
            return 0.1 / fd;
        }
        void updateQvaluesFromTGNN();  ///Ritika TGNN diff code
        void recordCqi(unsigned int sample, Direction dir);
        double getAverageCqi(Direction dir);
        double getVarianceCqi(Direction dir);


        double updateHysteresisTh(double v);
            double updateHysteresisThMinSinr(double v);
            double updateHysteresisThMinRsrp(double v);
            double updateHysteresisThMaxDist(double v);
            double updateHysteresisTowerLoad(double v);
            bool checkIfCellTowerPairExistsInMap( int t, double v);
            bool checkIfTowerExistsInMap( int t);
            void addToVC(double v, int t, std::vector<int>,NazaninHandoverDecision::SpeedCategory speedCategory,double scalarPar, double predictedValLSTM, double predictedValTGNN);

        //------------------------------
            struct CSVRow {
                double timestamp;
                  int vehicleId;
                  int masterId_;
                  int candidateMasterId_;
                  double signalQuality;
                  double masterDistance;
                  double candidateDistance;
                  double masterSpeed;
                  double candidateSpeed;
                  int vehicleDirection;
                  double towerLoad;
                  double vehiclePositionx;
                  double vehiclePositiony;
                  double vehiclePositionz;
                  double candidateTowerPositionx;
                  double candidateTowerPositiony;
                  double candidateTowerPositionz;
                  double masterRSSI;
                  double candidateRSSI;
                  double masterSINR;
                  double candidateSINR;
                  double masterRSRP;
                  double candidateRSRP;
                  double predictedLSTM;
                  double predictedTGNN;
                  int selectedTower;
            };
            void addRowToCSV(std::string& filename, const CSVRow& newRow);
            double calculateEachTowerLoad(int vehiclesConnectedToTower, int totalVehicles);
    // added by Mubashir
//    enum SpeedCategory {
//                   SPEED_020, SPEED_2140, SPEED_4160, SPEED_6180, SPEED_81100, SPEED_101120, SPEED_121140, SPEED_141160, SPEED_160PLUS, SPEED_COUNT
//               };
//    SpeedCategory getSpeedCategory(double vSpeed);
    void handlenormalHandover(double rssi, double rsrq, double maxSINR, double maxRSRP, double speedDouble, double distanceDouble, NazaninHandoverDecision::SpeedCategory speedCategory, bool isIntraHO);
    void updateCurrentMaster(double rssi, double maxSINR, double maxRSRP, double distanceDouble,double speedDouble);
    void handleFailureHandover(MacNodeId candidateID,double rssi, double rsrq, double maxSINR, double maxRSRP, double speedDouble, double distanceDouble,NazaninHandoverDecision::SpeedCategory speedCategory);
    void performHysteresisUpdate(double currentMasterRssi_, double currentMasterSinr_, double currentMasterRsrp_, double currentMasterDist_);

//    void calculateReward(double& rewd, double rssi, double avgLoad, double distanceDouble, std::vector<MacNodeId>& last_srv_MasterIdV);
    void calculateTimeInterval(double vIndiSpeed, double& srl_alpha, double& srl_gamma);
    void updateQValue(double& max_Qvalue, double& upt_Qvalue, std::vector<double>& upt_QvalueV, double ho_Qvalue,
                         double srl_alpha, double rewd, double srl_gamma, double srv_Qvalue, double mbr_Qvalue);
    void chooseAlgorithm(double scalPara, double predScaValLSTM, double predictedScaValTGNN, MacNodeId& sel_srv_Qvalue_id, MacNodeId& mbr_Qvalue_id, UserControlInfo* lteInfo);
    void handleHandoverDecision(LteAirFrame* frame, UserControlInfo* lteInfo, double rssi, double rsrq, double maxSINR, double maxRSRP,
            double distanceDouble, double speedDouble, double predScaValLSTM, double predScaValTGNN, NazaninHandoverDecision::SpeedCategory speedCategory);
    void performanceAnalysis();
    double minLoad = -10;

    std::vector<double>speedV;
    double sumSpeedV, avgSpeed;

    MacNodeId dirMacNodeId;

    //q value
    std::vector<double> upt_QvalueV;
    std::vector<double> avg_srv_QvalueV;
    double upt_Qvalue = 1, srv_Qvalue = 1, mbr_Qvalue = 1, ho_Qvalue = 1, rewd = 1, max_Qvalue = 1, avg_srv_Qvalue = 1;
    MacNodeId upt_Qvalue_id = 1, srv_Qvalue_id = 1, mbr_Qvalue_id = 1, ho_Qvalue_id = 1, max_Qvalue_id = 1, sel_srv_Qvalue_id = 1;
    double upt_srv_Qvalue = 1, upt_mbr_Qvalue = 1;
    double temp_upt_Qvalue = 1;
    double srl_alpha = 1, srl_gamma = 1;

    double ho_rssi = 10, ho_sinr = 10, ho_rsrp = -50, ho_dist = 1000, ho_load = 1;

    double insideVCHOVehicleTotal = 0, outsideVCHOVehicleTotal = 0,  failureHOVehicleTotal = 0, pingPongHOVehiclTotal = 0, vCurTowerVCHO = 0;
    double vvehicleCount020 = 0, vvehicleCount2140 = 0, vvehicleCount4160 = 0, vvehicleCount6180 = 0, vvehicleCount81100 = 0,
            vvehicleCount101120 = 0, vvehicleCount121140 = 0, vvehicleCount141160 = 0, vvehicleCount161 = 0, vvehicleCount160plus = 0;

    double vinsideHO020 = 0, fho_vinsideHO020 = 0, vinsideHO2140 = 0, fho_vinsideHO2140 = 0, vinsideHO4160 = 0, fho_vinsideHO4160 = 0,
            vinsideHO6180 = 0, fho_vinsideHO6180 = 0, vinsideHO81100 = 0, fho_vinsideHO81100 = 0, vinsideHO101120 = 0, fho_vinsideHO101120 = 0,
            vinsideHO121140 = 0, fho_vinsideHO121140 = 0, vinsideHO141160 = 0, fho_vinsideHO141160 = 0, vinsideHO160plus = 0, fho_vinsideHO160plus = 0;

    double vIndiSpeed = 0;

    double scalPara = 0, predScaValLSTM = 0, predScaValTGNN = 0;
    std::vector<int> closestTowerLst;
    std::tuple<double, double> predVehicleCoordSVR;
    double predXCoordVehicle;
    double predYCoordVehicle;

    MacNodeId oldMasterId_ = 1;
    std::vector<MacNodeId> last_srv_MasterIdV;

    std::string baseFilePath = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/";
    std::string speedFile = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/ChannelModel/speedFile.txt";

//    void calculatePredDirection(int& n, int& time1, int& time2, int& time3, int& time4, int& time5,
//                                              double& sumX, double& sumY, double& sumX2, double& sumXY, double& a, double& b,
//                                              double& tempdifCoordVTy, int& simtimeAdjstNum, int& lastTimeCount,
//                                              double& difCoordVTy, int& dirTower);


    double bsLoadCalWeighted;
//    const int NUM_TOWERS = 10;
//    double bsLoad[NUM_TOWERS] = {0};  // Use an array to store bsLoad values
//    double bsLoadCal[NUM_TOWERS] = {0};
//    double minLoad = 0.0;
//
//
//    enum SpeedCategory {
//        SPEED_020, SPEED_2140, SPEED_4160, SPEED_6180, SPEED_81100, SPEED_101120, SPEED_121140, SPEED_141160, SPEED_160PLUS, SPEED_COUNT
//    };
//
//
//
//    // Declare arrays for counts and values
//    std::array<double, SPEED_COUNT> vinsideHO;
//    std::array<double, SPEED_COUNT> fho_vinsideHO;
//    std::array<double, SPEED_COUNT> vehicleCount;
//    std::array<double, SPEED_COUNT> insideHO;
//    std::array<double, SPEED_COUNT> fho_insideHO;
//    std::array<double, SPEED_COUNT> pho_insideHO;
//    std::array<double, SPEED_COUNT> vvehicleCount;
//
//    SpeedCategory speedCategory;
//
//
//
   double minTowerLoad, avgLoad, sumbsLoadCal, towerCount;
    double towerLoad_cur_simtime = 1;
//
//    double countTotalHO, insideVCHO, fho_insideHOTotal, pho_insideHOTotal;
//
//    double vehicleCountTotal = 0;
//
//    double comHOTime = 0, insideVCHOTime = 0;
//    double vSpeed;
//    double updt_simtime = 1;
//    double finishSimTime = 1;
//    double lstmSimTime = 1;
//
//    int counter=0;
//    int counterLSTM=0;
//    int counterpredScalVal=0;
//
//    std::vector<double> inputLSTMTestDataArray;
//    std::vector<double> inputLSTMDataArray;
//    MacNodeId last_srv_MasterId_ = 1;

    double pre_srv_rssi = 0;


};

#endif  /* _LTE_AIRPHYUE_H_ */
