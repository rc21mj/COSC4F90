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

#include <assert.h>
#include "stack/phy/layer/LtePhyUe.h"

#include "stack/phy/layer/NRPhyUe.h"

#include "stack/ip2nic/IP2Nic.h"
#include "stack/phy/packet/LteFeedbackPkt.h"
#include "stack/phy/feedback/LteDlFeedbackGenerator.h"
#include "stack/phy/layer/NazaninHandoverDecision.h"

#include <ctime>
#include <vector>
#include <numeric>
#include <map>
#include <fstream>
#include <deque>
#include <sstream>

//#include <iostream>
//#include <sstream>
//#include <string>

std::deque<std::string> tgnnWindow_;
static const int TGNN_STEPS = 5;

int counter=0;
NazaninHandoverDecision* HoMng = new NazaninHandoverDecision();


//enum SpeedCategory {
//    SPEED_020, SPEED_2140, SPEED_4160, SPEED_6180, SPEED_81100, SPEED_101120, SPEED_121140, SPEED_141160, SPEED_160PLUS, SPEED_COUNT
//};

// Declare arrays for counts and values

std::array<double, NazaninHandoverDecision::SPEED_COUNT> insideHOVehicleLst;
std::array<double, NazaninHandoverDecision::SPEED_COUNT> outsideHOVehicleLst;
std::array<double, NazaninHandoverDecision::SPEED_COUNT> failureHOVehicleLst;
std::array<double, NazaninHandoverDecision::SPEED_COUNT> pingPongHOVehicleLst;
std::array<double, NazaninHandoverDecision::SPEED_COUNT> vehicleCountBySpeedLst;
std::array<double, NazaninHandoverDecision::SPEED_COUNT> virtualcellCountBySpeedLst;
//std::map<int,double> ho_Qvalue;  // candidateID -> Q-value  Ritika TGNN diff code



struct CSVRow {
    double timestamp;
      int vehicleId;
      int towerId;
      double signalQuality;
      double vehiclePositionx;
      double vehiclePositiony;
      double vehiclePositionz;
      double towerPositionx;
      double towerPositiony;
      double towerPositionz;
      double distance;
      double speed;
      int vehicleDirection;
      double towerLoad;
      int selectedTower;
};
std::string csvFilePath = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/simulator_data.csv";
bool isIntraHO;
double vc_cur_simtime = 1;
std::multimap<MacNodeId, double> VCid;  //storing vc id
std::multimap<MacNodeId, double>::iterator VCid_itr;
int currentVCSize;
double  totalVCSize; // total vc in the whole simulation
double testTotalClosestTowerFound, testHODecisionBySpeedCount, testHODecisionByScalarCount, totalScalarParaConditionedPassed = 0, testMasterIDCount =0;
int totalVehicleNoCreatedVC = 0; // number of vehicle vc is created for
double vehicleCountTotal = 0;
double countTotalHO, insideVCHOTotal, outsideVCHOTotal ,failureHOTotal, pingPongHOTotal;
double comHOTime = 0, insideVCHOTime = 0 , outsideVCHOTime =0, failureHOTime = 0;
double vSpeed;
double updt_simtime = 1, finishSimTime = 1, lstmSimTime = 1, lstChkVehicleId = 1;
bool isPerformedAnalysis = false;
std::vector<double> inputLSTMTestDataArray;
std::vector<double> inputLSTMDataArray;
std::vector<double> inputTGNNTestDataArray;   //Ritika
std::vector<double> inputTGNNDataArray;     //Ritika

std::unordered_set<int, int> countMap;

const int NUM_TOWERS = 10;
double tLoad[NUM_TOWERS] = {0};

Define_Module(LtePhyUe);

using namespace inet;

LtePhyUe::LtePhyUe()
{
    handoverStarter_ = nullptr;
    handoverTrigger_ = nullptr;
    cqiDlSum_ = cqiUlSum_ = 0;
    cqiDlCount_ = cqiUlCount_ = 0;
}

LtePhyUe::~LtePhyUe()
{
    cancelAndDelete(handoverStarter_);
    delete das_;
}

void LtePhyUe::initialize(int stage)
{
    LtePhyBase::initialize(stage);

    if (stage == inet::INITSTAGE_LOCAL)
    {
        isNr_ = true;  //was false        // this might be true only if this module is a NrPhyUe
        nodeType_ = UE;
        useBattery_ = false;  // disabled
        enableHandover_ = par("enableHandover");
        handoverLatency_ = par("handoverLatency").doubleValue();
        handoverDetachment_ = handoverLatency_/2.0;                      // TODO: make this configurable from NED
        handoverAttachment_ = handoverLatency_ - handoverDetachment_;
        dynamicCellAssociation_ = par("dynamicCellAssociation");
        if (par("minRssiDefault").boolValue())
            minRssi_ = binder_->phyPisaData.minSnr();
        else
            minRssi_ = par("minRssi").doubleValue();

        minSinr_ = 5;
        minRsrp_ = -90;
        maxDist_ = 2000;

        currentMasterRssi_ = 0;
        currentMasterSinr_ = 0;
        currentMasterRsrp_ = 0;
        candidateMasterDist_ = 0;
        candidateMasterSpeed_ = 0;

        currentMasterRssi_ = 0;
        candidateMasterSinr_ = 0;
        candidateMasterRsrp_ = 0;
        candidateMasterDist_ = 0;
        candidateMasterSpeed_ = 0;

        hysteresisTh_ = 0;
        hysteresisSinrTh_ = 0;
        hysteresisRsrpTh_ = 0;
        hysteresisDistTh_ = 0;
        hysteresisLoadTh_ = 0;

        hysteresisFactor_ = 10;
        hysteresisFactorSinr_ = 10;
        hysteresisFactorRsrp_ = 10;
        hysteresisFactorDist_ = 100;
        hysteresisFactorLoad_ = 1;
        handoverDelta_ = 0.00001;

        dasRssiThreshold_ = 1.0e-5;
        das_ = new DasFilter(this, binder_, nullptr, dasRssiThreshold_);

        servingCell_ = registerSignal("servingCell");
        averageCqiDl_ = registerSignal("averageCqiDl");
        averageCqiUl_ = registerSignal("averageCqiUl");

        if (!hasListeners(averageCqiDl_))
            error("no phy listeners");

        WATCH(nodeType_);
        WATCH(masterId_);
        WATCH(candidateMasterId_);
        WATCH(dasRssiThreshold_);
        WATCH(currentMasterRssi_);
        WATCH(currentMasterSinr_);
        WATCH(currentMasterRsrp_);
        WATCH(currentMasterDist_);
        WATCH(currentMasterSpeed_);
        WATCH(candidateMasterRssi_);
        WATCH(candidateMasterSinr_);
        WATCH(candidateMasterRsrp_);
        WATCH(candidateMasterDist_);
        WATCH(candidateMasterSpeed_);
        WATCH(hysteresisTh_);
        WATCH(hysteresisSinrTh_);
        WATCH(hysteresisRsrpTh_);
        WATCH(hysteresisDistTh_);
        WATCH(hysteresisLoadTh_);
        WATCH(hysteresisSpeedTh_);
        WATCH(hysteresisFactor_);
        WATCH(handoverDelta_);
    }
    else if (stage == inet::INITSTAGE_PHYSICAL_ENVIRONMENT)
    {
        if (useBattery_)
        {
            // TODO register the device to the battery with two accounts, e.g. 0=tx and 1=rx
            // it only affects statistics
            //registerWithBattery("LtePhy", 2);
//            txAmount_ = par("batteryTxCurrentAmount");
//            rxAmount_ = par("batteryRxCurrentAmount");
//
//            WATCH(txAmount_);
//            WATCH(rxAmount_);
        }

        txPower_ = ueTxPower_;

        lastFeedback_ = 0;

        handoverStarter_ = new cMessage("handoverStarter");

        if (isNr_)
        {
            mac_ = check_and_cast<LteMacUe *>(
                getParentModule()-> // nic
                getSubmodule("nrMac"));
            rlcUm_ = check_and_cast<LteRlcUm*>(
                getParentModule()-> // nic
                getSubmodule("nrRlc")->
                    getSubmodule("um"));
        }
        else
        {
            mac_ = check_and_cast<LteMacUe *>(
                getParentModule()-> // nic
                getSubmodule("mac"));
            rlcUm_ = check_and_cast<LteRlcUm*>(
                getParentModule()-> // nic
                getSubmodule("rlc")->
                    getSubmodule("um"));
        }
        pdcp_ = check_and_cast<LtePdcpRrcBase*>(
            getParentModule()-> // nic
            getSubmodule("pdcpRrc"));

        // get local id
        if (isNr_)
            nodeId_ = getAncestorPar("nrMacNodeId");
        else
            nodeId_ = getAncestorPar("macNodeId");
    }
    else if (stage == inet::INITSTAGE_PHYSICAL_LAYER)
    {
        // get serving cell from configuration
        // TODO find a more elegant way
        if (isNr_)
            masterId_ = getAncestorPar("nrMasterId");
        else
            masterId_ = getAncestorPar("masterId");
        candidateMasterId_ = masterId_;

        // find the best candidate master cell
        if (dynamicCellAssociation_)
        {
            // this is a fictitious frame that needs to compute the SINR
            LteAirFrame *frame = new LteAirFrame("cellSelectionFrame");
            UserControlInfo *cInfo = new UserControlInfo();

            // get the list of all eNodeBs in the network
            std::vector<EnbInfo*>* enbList = binder_->getEnbList();
            std::vector<EnbInfo*>::iterator it = enbList->begin();
            for (; it != enbList->end(); ++it)
            {
                // the NR phy layer only checks signal from gNBs
                if (isNr_ && (*it)->nodeType != GNODEB)
                    continue;

                // the LTE phy layer only checks signal from eNBs
                if (!isNr_ && (*it)->nodeType != ENODEB)
                    continue;

                MacNodeId cellId = (*it)->id;
                LtePhyBase* cellPhy = check_and_cast<LtePhyBase*>((*it)->eNodeB->getSubmodule("cellularNic")->getSubmodule("phy"));
                double cellTxPower = cellPhy->getTxPwr();
                Coord cellPos = cellPhy->getCoord();

                // TODO need to check if the eNodeB uses the same carrier frequency as the UE

                // build a control info
                cInfo->setSourceId(cellId);
                cInfo->setTxPower(cellTxPower);
                cInfo->setCoord(cellPos);
                cInfo->setFrameType(BROADCASTPKT);
                cInfo->setDirection(DL);

                //RSSI - Received Signal Strength Indicator
                //measurement of how well node can hear a signal from a tower
                // get RSSI from the eNB
                std::vector<double>::iterator it;
                double rssi = 0;
                std::vector<double> rssiV = primaryChannelModel_->getRSRP(frame, cInfo);
                for (it = rssiV.begin(); it != rssiV.end(); ++it){
                    rssi += *it;
                }

                rssi /= rssiV.size();   // compute the mean over all RBs

                if (rssi > candidateMasterRssi_)
                {
                    candidateMasterId_ = cellId;
                    srv_Qvalue_id = cellId;
                    candidateMasterRssi_ = rssi;
                }
            }
            delete cInfo;
            delete frame;

            // binder calls
            // if dynamicCellAssociation selected a different master
            if (candidateMasterId_ != 0 && candidateMasterId_ != masterId_)
            {
                binder_->unregisterNextHop(masterId_, nodeId_);
                binder_->registerNextHop(candidateMasterId_, nodeId_);
            }
            masterId_ = candidateMasterId_;
            // set serving cell
            if (isNr_)
                getAncestorPar("nrMasterId").setIntValue(masterId_);
            else
                getAncestorPar("masterId").setIntValue(masterId_);
                currentMasterRssi_ = candidateMasterRssi_;
                updateHysteresisTh(candidateMasterRssi_);
                if(currentMasterRssi_ > 2000){
                              // Ritika std::cout << "ShajibTest currentMasterRssi_  : " << masterId_ << " - masterId ID: " <<  currentMasterRssi_ << " is: "<< currentMasterRssi_ << std::endl;

                          }
        }

        das_->setMasterRuSet(masterId_);
        emit(servingCell_, (long)masterId_);
    }
    else if (stage == inet::INITSTAGE_NETWORK_CONFIGURATION)
    {
        // get cellInfo at this stage because the next hop of the node is registered in the IP2Nic module at the INITSTAGE_NETWORK_LAYER
        if (masterId_ > 0)
        {
            cellInfo_ = getCellInfo(nodeId_);
            int index = intuniform(0, binder_->phyPisaData.maxChannel() - 1);
            if (cellInfo_ != NULL)
            {
                cellInfo_->lambdaInit(nodeId_, index);
                cellInfo_->channelUpdate(nodeId_, intuniform(1, binder_->phyPisaData.maxChannel2()));
            }
        }
        else
            cellInfo_ = NULL;
    }
}

void LtePhyUe::handleSelfMessage(cMessage *msg)
{
//    std::cout << "(handleSelfMessage) -- msg: " << msg->getName() << std::endl;
    if (msg->isName("handoverStarter")){
//        std::cout << "(handleSelfMessage) -- handoverStarter" << msg << std::endl;
        triggerHandover();
    }

    else if (msg->isName("handoverTrigger"))
    {
//        std::cout << "BEFORE (doHandover) -- master id:" << masterId_ << " candidate master id: "<<candidateMasterId_<< std::endl;
        doHandover();
        delete msg;
        handoverTrigger_ = nullptr;
//        std::cout << "AFTER (doHandover) -- master id:" << masterId_ << " candidate master id: "<<candidateMasterId_<< std::endl;
    }
}

void LtePhyUe::addRowToCSV(std::string& filename, const CSVRow& newRow) {
  std::fstream file(filename, std::ios::out | std::ios::app);

  if (!file.is_open()) {
      std::cerr << "Error:addRowToCSV Unable to open the file for writing." << std::endl;
      return;
  }


  // Write the new row to the file /////Ritika added newRow.predictedTGNN
  file << newRow.timestamp << ',' << newRow.vehicleId << ',' << newRow.masterId_ << ','<< newRow.candidateMasterId_ << ',' << newRow.signalQuality
          << ',' << newRow.masterDistance << ',' << newRow.candidateDistance << ','<< newRow.masterSpeed << ','<< newRow.candidateSpeed <<  ',' << newRow.vehicleDirection << ',' << newRow.vehiclePositionx
          <<',' <<newRow.vehiclePositiony << ','<< newRow.vehiclePositionz <<','<<newRow.candidateTowerPositionx << ','<< newRow.candidateTowerPositiony
          <<',' <<newRow.candidateTowerPositionz << ',' << newRow.towerLoad << ',' << newRow.masterRSSI << ',' << newRow.candidateRSSI << ',' << newRow.masterSINR << ',' << newRow.candidateSINR
          << ',' << newRow.masterRSRP   << ',' << newRow.candidateRSRP << ','  << newRow.predictedTGNN << ',' << newRow.predictedLSTM << ',' << newRow.selectedTower <<  '\n' ;
  file.close();
}

////Ritika TGNN diff code

void LtePhyUe::updateQvaluesFromTGNN()
{
    std::ifstream infile("/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/outputTGNNdiff.txt");
    if (!infile.is_open()) {
        EV << "Error: Cannot open TGNN output file!\n";
        return;
    }

    double tgnn_score = 0.0;
        infile >> tgnn_score;
        infile.close();

    std::cout << "[TGNNdiff READ] simTime=" << simTime()
            << " read tgnn_score=" << tgnn_score
            << " from outputTGNNdiff.txt\n" << std::endl;

    // Assign directly
    double oldQ = ho_Qvalue;
    ho_Qvalue = tgnn_score;

    std::cout << "[TGNNdiff APPLY] simTime=" << simTime()
            << " ho_Qvalue was=" << oldQ
            << " now=" << ho_Qvalue << "\n" << std::endl;


}    ////Ritika TGNN diff code

double LtePhyUe::calculateEachTowerLoad(int vehiclesConnectedToTower, int totalVehicles) {
    // Check if the total number of vehicles is greater than zero to avoid division by zero
    if (totalVehicles > 0) {
        return static_cast<double>(vehiclesConnectedToTower) / totalVehicles * 100.0;
    } else {
        // Handle the case where the total number of vehicles is zero
        std::cerr << "Error: Total number of vehicles is zero." << std::endl;
        return -1.0; // Return a special value to indicate an error
    }
}

//Ritika functions
void LtePhyUe::appendProperTGNNRow(const ProperTGNNRow& row)
{
    properTGNNWindow_.push_back(row);

    while ((int)properTGNNWindow_.size() > PROPER_TGNN_STEPS) {
        properTGNNWindow_.pop_front();
    }
}

bool LtePhyUe::writeProperTGNNWindowToFile(const std::string& filepath)
{
    if ((int)properTGNNWindow_.size() < PROPER_TGNN_STEPS) {
        EV_WARN << "[ProperTGNN] Not enough rows to write runtime window. size="
                << properTGNNWindow_.size() << " required=" << PROPER_TGNN_STEPS << endl;
        return false;
    }

    std::ofstream out(filepath.c_str());
    if (!out.is_open()) {
        EV_ERROR << "[ProperTGNN] Failed to open runtime window file: " << filepath << endl;
        return false;
    }

    out << "timestamp,vehicleId,masterId,candidateMasterId,"
           "masterDistance,candidateDistance,"
           "masterSpeed,candidateSpeed,"
           "vehicleDirection,"
           "vehiclePosition-x,vehiclePosition-y,"
           "towerload,"
           "masterRSSI,candidateRSSI,"
           "masterSINR,candidateSINR,"
           "masterRSRP,candidateRSRP\n";

    for (const auto& r : properTGNNWindow_) {
        out << r.timestamp << ","
            << r.vehicleId << ","
            << r.masterId << ","
            << r.candidateMasterId << ","
            << r.masterDistance << ","
            << r.candidateDistance << ","
            << r.masterSpeed << ","
            << r.candidateSpeed << ","
            << r.vehicleDirection << ","
            << r.vehiclePosX << ","
            << r.vehiclePosY << ","
            << r.towerload << ","
            << r.masterRSSI << ","
            << r.candidateRSSI << ","
            << r.masterSINR << ","
            << r.candidateSINR << ","
            << r.masterRSRP << ","
            << r.candidateRSRP
            << "\n";
    }

    out.close();

    EV_INFO << "[ProperTGNN] Wrote runtime window with "
            << properTGNNWindow_.size() << " rows to " << filepath << endl;

    return true;
}


////////////////////////////////////////////////////////
//only called in (NRPhyUe::handleAirFrame)
void LtePhyUe::handoverHandler(LteAirFrame* frame, UserControlInfo* lteInfo)
{
    counter++;
    if (counter==1){
        //clearing the content of the csv file from previous run
        std::ofstream file(csvFilePath, std::ios::out | std::ios::trunc);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open the file for clearing." << std::endl;
            return;
        }
        file.close();

        // opening the csv file to write the new data
        std::fstream file1(csvFilePath, std::ios::out | std::ios::app);
      if (!file1.is_open()) {
          std::cerr << "Error:handoverHandler LtePhyUe Unable to open the file for writing." << std::endl;
          return;
      }
      // Write the CSV column names /////Ritika added predictedTGNN
      file1 << "timestamp" << ',' <<"vehicleId" << ',' << "masterId" << ',' << "candidateMasterId" << ',' << "signalQuality"
           << ',' << "masterDistance"  << ',' << "candidateDistance" << ',' << "masterSpeed" << ',' << "candidateSpeed"
           << ',' << "vehicleDirection" << ',' << "vehiclePosition- x" << ',' << "vehiclePosition- y" <<','<<"vehiclePosition- z"
           << ',' << "candidateTowerPosition- x" <<','<<"candidateTowerPosition- y"<<','<<"candidateTowerPosition- z"<<','<< "towerload" << ','<< "masterRSSI" << ',' << "candidateRSSI"
            << ',' << "masterSINR" << ',' << "candidateSINR" << ',' << "masterRSRP"<< ',' << "candidateRSRP" << ',' << "predictedTGNN" << ',' << "predictedLSTM" << ','<< "selectedTower" << '\n';
      file1.close();
    }

    lteInfo->setDestId(nodeId_);
//    std::cout << "--------------------------------------------------------------------------" << std::endl;
//    std::cout << "COUNTER: " << counter << " ,VEHICLE ID: " << lteInfo->getDestId() << ", TOWER ID: " << lteInfo->getSourceId() << std::endl;
//    std:: cout << "masterID: " << masterId_ << std::endl;

    if (!enableHandover_)
    {
        // Even if handover is not enabled, this call is necessary
        // to allow Reporting Set computation.
        if (getNodeTypeById(lteInfo->getSourceId()) == ENODEB && lteInfo->getSourceId() == masterId_)
        {
            // Broadcast message from my master enb
            das_->receiveBroadcast(frame, lteInfo);
        }

        delete frame;
        delete lteInfo;
        return;
    }

    frame->setControlInfo(lteInfo);


//    calculate RSSI, sinr, rsrp, rsrq
    auto result = HoMng->calculateMetrics(primaryChannelModel_, frame, lteInfo);

    // Access the values from the tuple
    double rssi = std::get<0>(result);
    double maxSINR = std::get<1>(result);
    double maxRSRP = std::get<2>(result);
    double rsrq = std::get<3>(result);

    //std::cout << "Shajib(OUTSIDE) Check calculateMetrics call counter" << counter << "RSSI"<<  rssi << " maxSINR " << maxSINR << " maxRSRP " << maxRSRP << endl;
//    std::cout << "handoverHandler calculatemetrics: " << std::endl;

//    std::cout << "RSSI: " << rssi << std::endl;
//    std::cout << "maxSINR: " << maxSINR << std::endl;
//    std::cout << "maxRSRP: " << maxRSRP << std::endl;
//    std::cout << "rsrq: " << rsrq << std::endl;


    // Getting the distance from vehicle and tower from NRPhyUe.cc
    double distanceDouble = HoMng->getParfromFile(baseFilePath+"distanceFile.txt");
 //   std::cout << "VEHICLE ID: " << nodeId_ << " - TOWER ID: " <<  lteInfo->getSourceId() << " DISTANCE btw is: "<< distanceDouble << std::endl;

    // Getting the vehicle position from NRPhyUe.cc
    std::ifstream readfile(baseFilePath+"vehiclePositionFile.txt");
    double vPositionx, vPositiony, vPositionz;
    readfile >> vPositionx >> vPositiony >> vPositionz;
    readfile.close();
    //std::cout << "VEHICLE ID: " << nodeId_ << " POSITION: x: " << vPositionx << " y: " <<vPositiony << std::endl;


    Coord towerPosition = lteInfo->getCoord();
//    std::cout << "TOWER ID: " << lteInfo->getSourceId()  << " POSITION: "<< towerPositionDouble << std::endl;

    //speed calculation using file from LteRealisticChannelModel::computeSpeed
    double speedDouble = HoMng->getParfromFile("/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/ChannelModel/speedFile.txt");
    //std::cout << "VEHICLE ID: "<< nodeId_ <<", Check SPEED: " << speedDouble << std::endl;
    double vSpeed = speedDouble * 3.6; //converting to km/h
    vIndiSpeed = vSpeed;

    speedV.push_back(speedDouble);

        //calculate tower load
     HoMng->calculateTowerLoad(lteInfo, frame);
     int connectedVehicles= HoMng->bsLoad[lteInfo->getSourceId()-1];
     double towerload = calculateEachTowerLoad(connectedVehicles,2000);
    // std::cout << "TOWER ID: " << lteInfo->getSourceId() << " LOAD: "<< towerload<< std::endl;
     avgLoad = towerload;

     // (dirFile.txt) Saved in NRPhyUe.cc initialize()
     dirMacNodeId = HoMng->getParfromFile(baseFilePath+"dirFile.txt");
     // std::cout << "DIRECTION " << dirMacNodeId << std::endl;

     // (comHOTimeCount.txt) Saved in NRPhyUe.cc initialize()
     comHOTime = HoMng->getParfromFile(baseFilePath+"comHOTimeCount.txt");

     //Vehicle count (nodeCount.txt) Saved in LteMacUe.cc initialize()
     vehicleCountTotal = HoMng->getParfromFile(baseFilePath+"nodeCount.txt");
     //std::cout << "vehicleCount: " << vehicleCountTotal << endl;

     //calculate scalar format of parameters
     //scalPara = ((rssi + maxSINR + maxRSRP + avgLoad + distanceDouble) / 5);;
     scalPara = rssi;
     //lstm input3
     inputTGNNDataArray.push_back(scalPara);  ///Ritika
     inputLSTMDataArray.push_back(scalPara);

     ///// Ritika TGNN diss code
     std::stringstream ss;
     ss << simTime().dbl() << " "
        << getMacNodeId() << " "
        << rssi << " "
        << maxSINR << " "
        << speedDouble << " "
        << distanceDouble;

     tgnnWindow_.push_back(ss.str());

     if (tgnnWindow_.size() > TGNN_STEPS) {
         tgnnWindow_.pop_front();
     }
    ///// ritika TGNN diff code ended

     //std::cout << "RSSI: " << rssi << " is pushed to inputLSTM data." << std::endl;
     //std::cout << "length of inputLSTMDataArray: " << inputLSTMDataArray.size()  << std::endl;

/////////////Ritika

     {
         ProperTGNNRow properRow;

         properRow.timestamp = simTime().dbl();
         properRow.vehicleId = getMacNodeId();
         properRow.masterId = masterId_;
         properRow.candidateMasterId = candidateMasterId_;

         properRow.masterDistance = currentMasterDist_;
         properRow.candidateDistance = distanceDouble;

         properRow.masterSpeed = currentMasterSpeed_;
         properRow.candidateSpeed = speedDouble;   // replace if you have a true candidate speed
         properRow.vehicleDirection = dirMacNodeId;

         properRow.vehiclePosX = vPositionx;
         properRow.vehiclePosY = vPositiony;

         properRow.towerload = towerload;

         properRow.masterRSSI = currentMasterRssi_;
         properRow.candidateRSSI = rssi;

         properRow.masterSINR = currentMasterSinr_;
         properRow.candidateSINR = maxSINR;

         properRow.masterRSRP = currentMasterRsrp_;
         properRow.candidateRSRP = maxRSRP;

         appendProperTGNNRow(properRow);

         if ((int)properTGNNWindow_.size() == PROPER_TGNN_STEPS) {
             std::string properInputFile = baseFilePath + "runtime_tgnn_window.csv";
             writeProperTGNNWindowToFile(properInputFile);
         }
     }


       if (((int)simTime().dbl() % 13 == 0) || ((int)simTime().dbl() % 14 == 0))
       {
           inputLSTMTestDataArray.push_back(scalPara);
           inputTGNNTestDataArray.push_back(scalPara);       //Ritika
           HoMng->saveArrayToFile("inputLSTMTestData.txt", inputLSTMTestDataArray);
           HoMng->saveArrayToFile("inputLSTM.txt", inputLSTMDataArray);
           HoMng->saveArrayToFile("inputTGNNTestData.txt", inputTGNNTestDataArray);   //Ritika
           HoMng->saveArrayToFile("inputTGNN.txt", inputTGNNDataArray);               //Ritika
       }

       //lstm run
       if (((int)simTime().dbl() % 15 == 0) && (simTime().dbl() != lstmSimTime))
       {

           HoMng->runLSTM();
           HoMng->runTGNN();   //Ritika
           HoMng->runTGNNdiff();  ///Ritika TGNN diff code
           HoMng->runProperTGNN();  ///Ritika TGNN diff code
           lstmSimTime = simTime().dbl();
//           predScaValLSTM = HoMng->getParfromFile(baseFilePath+"outputLSTM.txt"); // Update for each vehicle separately. as "predScaValLSTM" is declared in .h file
       }
       int after_5SimTime = ((int)simTime().dbl()+5);

       if (((int)simTime().dbl() % 15 == 0) && nodeId_ !=lstChkVehicleId )
       {
            HoMng->runSVR(nodeId_,after_5SimTime);
            //std::cout << "Shajib:: SVR CALL AT: " << (int)simTime().dbl() << endl;
            //HoMng->runLSTM();
             lstChkVehicleId = nodeId_;
             predVehicleCoordSVR = HoMng->getParfromFileForSVR(baseFilePath + "outputSVR.txt" );
             predXCoordVehicle = std::get<0>(predVehicleCoordSVR);
             predYCoordVehicle = std::get<1>(predVehicleCoordSVR);
             closestTowerLst= HoMng->GetClosestTowersId(predXCoordVehicle, predYCoordVehicle, nodeId_, lteInfo->getSourceId() );
        }
       if ((int)simTime().dbl() % 16 == 0)
       {
           HoMng->saveParaToFile("inputLSTM.txt", 0);
           HoMng->saveParaToFile("inputLSTMTestData.txt", 0);
           inputLSTMTestDataArray.clear();
           inputLSTMDataArray.clear();
           HoMng->saveParaToFile("inputTGNN.txt", 0);     //Ritika
           HoMng->saveParaToFile("inputTGNNTestData.txt", 0);  //Ritika
           inputTGNNTestDataArray.clear();        //Ritika
           inputTGNNDataArray.clear();            //Ritika
       }

       //lstm output
       predScaValLSTM = HoMng->getParfromFile(baseFilePath+"outputLSTM.txt");
       predScaValTGNN = HoMng->getParfromFile(baseFilePath+"outputTGNN.txt");     ///Ritika
      // std::cout<<" Shajib:: predScaValLSTM for vehicle " << nodeId_ << " is " << predScaValLSTM << " At time: "<< (int)simTime().dbl()<<  endl;
     //  auto  predVehicleCoordSVR = HoMng->getParfromFileForSVR(baseFilePath + "outputSVR.txt" );
//       double predXCoordVehicle = std::get<0>(predVehicleCoordSVR);
//       double predYCoordVehicle = std::get<1>(predVehicleCoordSVR);
      // std::cout<<" Shajib:: predXCoord for vehicle " << nodeId_ << " is X:" << predXCoordVehicle << " and Y:"<< predYCoordVehicle << " At time: "<< (int)simTime().dbl() << "TOTAL closestTowerLst "<< closestTowerLst.size()<<  endl;


      // std::vector<int> closestTowerLst= HoMng->GetClosestTowersId(predXCoordVehicle, predYCoordVehicle, nodeId_, lteInfo->getSourceId() );


       //distance between vehicle and tower
       //double sqrDistance = otherPhy_->getCoord().distance(lteInfo->getCoord());
       //std::cout << "Shajib:: Distance Of Tower: " << lteInfo->getSourceId()  << " and  Vehicle" <<  distanceDouble << " is: "<<   sqrDistance << "NRPHY FIle Distance: " << distanceDouble << std::endl;
       //std::cout << "Shajib:: Distance Of Tower: " << lteInfo->getSourceId()  << " and  Vehicle" <<  nodeId_ << " is: "<<   distanceDouble<< std::endl;


       //Count the number of vehicles based on their speed ranges
       NazaninHandoverDecision::SpeedCategory speedCategory = HoMng->getSpeedCategory(vSpeed);
       vehicleCountBySpeedLst[speedCategory]++;
       //Virtual CELL STORE
       //addToVC(nodeId_, lteInfo->getSourceId(), closestTowerLst, speedCategory, scalPara, predScaValLSTM);
       addToVC(nodeId_, lteInfo->getSourceId(), closestTowerLst, speedCategory, scalPara, predScaValLSTM, predScaValTGNN);  //Ritika


        //A reward (rewd) is calculated based on (rssi), average load (avgLoad), and the distance between the vehicle and the base station.
        HoMng->calculateReward(rewd, rssi, avgLoad, distanceDouble, last_srv_MasterIdV);

        // Clear a list of last serving master IDs every 2 simulation time units
        if ((int)simTime().dbl() % 2 == 0) {
            last_srv_MasterIdV.clear();
        }

        //Set speed-related parameters (alpha and gamma) based on vehicle speed ranges
        HoMng->calculateTimeInterval(vIndiSpeed,srl_alpha, srl_gamma);

        //upt_Qvalue is calculated using a formula involving several Q-values (ho_Qvalue, srv_Qvalue, mbr_Qvalue), rewards, and scaling factors
        //HoMng->updateQValue(max_Qvalue, upt_Qvalue, upt_QvalueV, ho_Qvalue, srl_alpha, rewd, srl_gamma, srv_Qvalue, mbr_Qvalue);

        //HoMng->chooseAlgorithm(scalPara, predScaValLSTM,sel_srv_Qvalue_id,  mbr_Qvalue_id, lteInfo);


        if (candidateMasterRssi_ > predScaValTGNN) {      //Ritika
            sel_srv_Qvalue_id = lteInfo->getSourceId();
            //std::cout << "scalPara > predScaValLSTM" << std::endl;
        } else {
            mbr_Qvalue_id = lteInfo->getSourceId();
            //std::cout << "scalPara < predScaValLSTM" << std::endl;
        }

        //std::cout<< "sel_srv_Qvalue_id: " << sel_srv_Qvalue_id << ", mbr_Qvalue_id: "<< mbr_Qvalue_id << std::endl;
        //inserting direction prediction
        //vc_set.insert(dirMacNodeId);

       //bd msg from the serving tower (already connected)
       if (getNodeTypeById(lteInfo->getSourceId()) == ENODEB && lteInfo->getSourceId() == masterId_)
       {
           //Code for processing broadcast message from the master gNB
           rssi = das_->receiveBroadcast(frame, lteInfo);
           double sqrDistance = lteInfo->getCoord().distance(lteInfo->getCoord());
           rsrq = (10 * maxRSRP) / rssi;

           distanceDouble = HoMng->getParfromFile(baseFilePath+"distanceFile.txt");
           if(distanceDouble > 2000){
               // Ritika std::cout << "ShajibTest DISTANCE btw VEHICLE ID: " << nodeId_ << " - TOWER ID: " <<  lteInfo->getSourceId() << " is: "<< distanceDouble << std::endl;

           }
           //std::cout << "DISTANCE btw VEHICLE ID: " << nodeId_ << " - TOWER ID: " <<  lteInfo->getSourceId() << " is: "<< distanceDouble << std::endl;

           speedDouble = HoMng->getParfromFile("/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/ChannelModel/speedFile.txt");
           //std::cout << "VEHICLE ID: "<< nodeId_ <<", SPEED: " << speedDouble << std::endl;

           avg_srv_QvalueV.push_back(upt_Qvalue);
           avg_srv_Qvalue = std::accumulate(avg_srv_QvalueV.begin(), avg_srv_QvalueV.end(), 0.0) / avg_srv_QvalueV.size();

           last_srv_MasterIdV.push_back(masterId_);
           srv_Qvalue = rsrq + (200 - maxSINR);
       }
       //bd msg from other towers
       else
       {
           //Code for processing broadcast messages from other (not-master) gNBs
           //Calculate RSSI based on SINR values
           auto result = HoMng->calculateMetrics(primaryChannelModel_, frame, lteInfo);

           // Access the values from the tuple
           rssi = std::get<0>(result);
           if(rssi > 2000){
                                // Ritika    std::cout << "ShajibTest rssi  : " << rssi << " - masterId ID: " <<  masterId_ <<std::endl;

                                }
           maxSINR = std::get<1>(result);
           maxRSRP = std::get<2>(result);
           rsrq = std::get<3>(result);

           //std::cout << "Shajib(Inside) Check calculateMetrics call counter" << counter << "RSSI"<<  rssi << " maxSINR " << maxSINR << " maxRSRP " << maxRSRP << endl;
           speedDouble = HoMng->getParfromFile("/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/ChannelModel/speedFile.txt");
           //std::cout << "VEHICLE ID: "<< nodeId_ <<", SPEED: " << speedDouble << std::endl;

           mbr_Qvalue = rsrq + (200 - maxSINR);
       }

       if (lteInfo->getSourceId() != masterId_ && rssi < minRssi_) //signal is low to consider tower
       {
           delete frame;
           return;
       }

       //Shajib If vehicle Id is in Map () Add in VCID

       //////////////////////////////Ritika TGNN diff code
       if (tgnnWindow_.size() == TGNN_STEPS) {

                  const std::string tgnnTestPath =
                      "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/"
                      "simu5G/src/stack/phy/layer/inputTGNNdiffTestData.txt";

                  std::ofstream testFile(tgnnTestPath, std::ios::trunc);
                  if (!testFile.is_open()) {
                      EV << "Error: Cannot open TGNN diff test file for writing: " << tgnnTestPath << "\n";
                  } else {
                      for (const auto& line : tgnnWindow_) {
                          testFile << line << std::endl;
                      }
                      testFile.close();

                      // Run the TGNN-diff python and then load its output
                      HoMng->runTGNNdiff();
                      updateQvaluesFromTGNN();
                  }
         }
       /////Ritika TGNN diff code

       //handover decision
       //std::cout << "Shajib::handoverhandler Vehicle: "<< nodeId_ << " CurTower: "<< lteInfo->getSourceId() << " masterId_ " << masterId_ << " CandidateMasterID "<< candidateMasterId_ << " RSSI VALUE:  "<< rssi << " candidateMasterRssi: " << candidateMasterRssi_ << " hysteresisTh_ : "<<  hysteresisTh_<< " (candidateMasterRssi_ + hysteresisTh_): " << (candidateMasterRssi_ + hysteresisTh_) << " predScaValLSTM: " << predScaValLSTM << endl;
       //std::cout<< "Scalar: " << scalPara << " predScaValLSTM: " << predScaValLSTM  << " candidateMasterRssi_ " << candidateMasterRssi_<< endl;
       MacNodeId selectedTower = 0;
       if (predScaValTGNN >  candidateMasterRssi_ + hysteresisTh_) {   //Ritika

           if(lteInfo->getSourceId() == masterId_){
               testMasterIDCount++;
           }
           totalScalarParaConditionedPassed++;
           //We have a good signal quality even better than predicted
           if (sel_srv_Qvalue_id == masterId_) {

               // receiving even stronger broadcast from current master
               updateCurrentMaster(rssi, maxSINR, maxRSRP, distanceDouble,speedDouble);
               candidateMasterId_ = masterId_;
               oldMasterId_ = masterId_;
               performHysteresisUpdate(currentMasterRssi_, currentMasterSinr_, currentMasterRsrp_, currentMasterDist_);
               cancelEvent(handoverStarter_);
           } else {
               //receiving stronger broadcast from another non master tower -> schedule a handover to it -> set it as candidate master
               if(checkIfTowerExistsInMap(lteInfo->getSourceId())){
                   //std::cout << "Shajib::checkIfTowerExistsInMap YES! Tower " << lteInfo->getSourceId() << "Found" << endl;
                   isIntraHO = true;
               }
               else{
                   isIntraHO = false;
               }
               selectedTower = sel_srv_Qvalue_id;
               handlenormalHandover(rssi,rsrq, maxSINR, maxRSRP, speedDouble, distanceDouble, speedCategory,isIntraHO);
               testHODecisionByScalarCount++;
           }
       }
       //Even If scalPara is less than predScaValLSTM. Check the (Vehicle, Tower) pair in the MAP. If Exists -> It is close to Tower. Can do handover.
       else if((masterId_ != lteInfo->getSourceId()) && checkIfCellTowerPairExistsInMap(lteInfo->getSourceId(),nodeId_ )   ){
           //std::cout << "Shajib::checkIfCellTowerPairExistsInMap YES!: Vehicle " << nodeId_ << " CURTower: "<< lteInfo->getSourceId()  << endl;
           //inTraVC handover
           //std::cout << "Shajib:: Checking Vehicle "<< nodeId_ << " and Tower "<< lteInfo->getSourceId() << " scalPara: "<< scalPara << " predScaValLSTM: " << predScaValLSTM <<  " sel_srv_Qvalue_id: "<< sel_srv_Qvalue_id << " masterId_: "<< masterId_ << " oldMasterId_: "<< oldMasterId_ << " predXCoordVehicle: " << predXCoordVehicle << " predYCoordVehicle: "<< predYCoordVehicle << " distanceDouble: "<< distanceDouble<< endl;
           sel_srv_Qvalue_id = lteInfo->getSourceId();
           isIntraHO = true;
           selectedTower = sel_srv_Qvalue_id;
           handlenormalHandover(rssi,rsrq, maxSINR, maxRSRP, speedDouble, distanceDouble, speedCategory,isIntraHO);
           //performHysteresisUpdate(currentMasterRssi_, currentMasterSinr_, currentMasterRsrp_, currentMasterDist_);
           testHODecisionBySpeedCount++;
       }
       else{
           if (lteInfo->getSourceId() == masterId_)
           {
               if (rssi >= minRssi_)
               {
                   currentMasterRssi_ = rssi;
                   candidateMasterRssi_ = rssi;
                   hysteresisTh_ = updateHysteresisTh(rssi);
                   // Can use  updateCurrentMaster()
               }
               else  // lost connection from current master
               {
                   if (candidateMasterId_ == masterId_)  // trigger detachment
                   {
                       candidateMasterId_ = 0;
                       candidateMasterRssi_ = 0;
                       hysteresisTh_ = updateHysteresisTh(0);
                       handleFailureHandover(0,0,0, 0, maxRSRP, 0, 0,speedCategory);
                       //Can use performHysteresisUpdate()
                   }
                   // else do nothing, a stronger RSSI from another nodeB has been found already
               }
           }
       }
//       if(masterId_!= candidateMasterId_ & selectedTower!= sel_srv_Qvalue_id){
//           selectedTower = 999;
//       }

    CSVRow newRow;

    // Set values for the members of the CSVRow struct
//    newRow.vehicleId = nodeId_;
//    //newRow.towerId = lteInfo->getSourceId();
//    newRow.masterId_ = masterId_;
//    newRow.candidateMasterId_ = candidateMasterId_;
//    newRow.signalQuality = rssi;
//    newRow.distance = distanceDouble;
//    newRow.speed = speedDouble;
//    newRow.vehicleDirection = dirMacNodeId;
//    newRow.towerLoad = towerload;
//    newRow.timestamp = simTime().dbl();
//    newRow.masterTowerPositionx = towerPosition.x;
//    newRow.masterTowerPositiony = towerPosition.y;
//    newRow.masterTowerPositionz = towerPosition.z;
//    newRow.vehiclePositionx = vPositionx;
//    newRow.vehiclePositiony = vPositiony;
//    newRow.vehiclePositionz = vPositionz;
//    newRow.rssi = rssi;
//    newRow.predictedLSTM = predScaValLSTM;
//    newRow.selectedTower = selectedTower;
//
//    currentMasterRssi_ = 0;
//    currentMasterSinr_ = 0;
//    currentMasterRsrp_ = 0;
//    candidateMasterDist_ = 0;
        newRow.timestamp = simTime().dbl();
        newRow.vehicleId = nodeId_;
        newRow.masterId_ = masterId_;
        newRow.candidateMasterId_ = candidateMasterId_;
        newRow.signalQuality = rssi;
        newRow.masterDistance = currentMasterDist_;
        newRow.candidateDistance = distanceDouble;
        newRow.masterSpeed = currentMasterSpeed_;
        newRow.candidateSpeed = speedDouble;
        newRow.vehicleDirection = dirMacNodeId;
       // newRow.towerLoad = towerload;

//        newRow.masterTowerPositionx = towerPosition.x;
//        newRow.masterTowerPositiony = towerPosition.y;
//        newRow.masterTowerPositionz = towerPosition.z;
        newRow.candidateTowerPositionx = towerPosition.x;
        newRow.candidateTowerPositiony = towerPosition.y;
        newRow.candidateTowerPositionz = towerPosition.z;
        newRow.vehiclePositionx = vPositionx;
        newRow.vehiclePositiony = vPositiony;
        newRow.vehiclePositionz = vPositionz;
        newRow.masterRSSI = currentMasterRssi_;
        newRow.candidateRSSI = rssi;
        newRow.masterSINR = maxSINR;
        newRow.candidateSINR = currentMasterSinr_;
        newRow.masterRSRP = currentMasterRsrp_;
        newRow.candidateRSRP = maxRSRP;
        newRow.predictedLSTM = predScaValLSTM;
        newRow.predictedTGNN = predScaValTGNN;
        newRow.selectedTower = selectedTower;
    // Add the new row to the CSV file
    addRowToCSV(csvFilePath, newRow);
    //performance analysis
    performanceAnalysis();

    delete frame;
}
/////////////////////////////////////////////////////////////////

void LtePhyUe::triggerHandover()
{
    //do a flush here
   // std::cout << "triggerHandover method -> handover from: " << masterId_ << "to: " << candidateMasterId_ << std::flush;

//    ASSERT(masterId_ != candidateMasterId_);
    //the candidate master ID is the one we're doing the handover to and if it is the same as master id then we shouldn't do the handover because we're already connected to it.

//    std::cout << "##### Handover Starting #####" << endl;
//    std::cout << "Current Master ID: " << masterId_ << endl;
//    std::cout << "Candidate Master ID: " << candidateMasterId_ << endl;
//    std::cout << "##########" << endl;

    if (candidateMasterRssi_ == 0) {
        // EV << NOW << "LtePhyUe::triggerHandover - UE " << nodeId_ << " lost its connection to gNB " << masterId_ << ". Now detaching... " << endl;
//        std::cout << NOW << "LtePhyUe::triggerHandover - UE " << nodeId_ << " lost its connection to gNB " << masterId_ << ". Now detaching... " << endl;
    }

    else if (masterId_ == 0) {
        // EV << NOW << " LtePhyUe::triggerHandover - UE " << nodeId_ << " is starting attachment procedure to gNB " << candidateMasterId_ << "... " << endl;
//        std::cout << NOW << " LtePhyUe::triggerHandover - UE " << nodeId_ << " is starting attachment procedure to gNB " << candidateMasterId_ << "... " << endl;
    }
    else {
        // EV << NOW << " LtePhyUe::triggerHandover - UE " << nodeId_ << " is starting handover to gNB " << candidateMasterId_ << "... " << endl;
//        std::cout << NOW << " LtePhyUe::triggerHandover - UE " << nodeId_ << " is starting handover to gNB " << candidateMasterId_ << "... " << endl;
    }
    binder_->addUeHandoverTriggered(nodeId_);

    // inform the UE's IP2Nic module to start holding downstream packets
    IP2Nic* ip2nic =  check_and_cast<IP2Nic*>(getParentModule()->getSubmodule("ip2nic"));
    ip2nic->triggerHandoverUe(candidateMasterId_);
    binder_->removeHandoverTriggered(nodeId_);

    // inform the eNB's IP2Nic module to forward data to the target eNB
    if (masterId_ != 0 && candidateMasterId_ != 0)
    {
        IP2Nic* enbIp2Nic =  check_and_cast<IP2Nic*>(getSimulation()->getModule(binder_->getOmnetId(masterId_))->getSubmodule("cellularNic")->getSubmodule("ip2nic"));
        enbIp2Nic->triggerHandoverSource(nodeId_,candidateMasterId_);
    }

    double handoverLatency;
    if (masterId_ == 0)                        // attachment only
        handoverLatency = handoverAttachment_;
    else if (candidateMasterId_ == 0)          // detachment only
        handoverLatency = handoverDetachment_;
    else
        handoverLatency = handoverDetachment_ + handoverAttachment_;

    handoverTrigger_ = new cMessage("handoverTrigger");
    scheduleAt(simTime() + handoverLatency, handoverTrigger_);
}

void LtePhyUe::doHandover()
{

    std::cout << "doHandover method -> handover from: " << masterId_ << "to: " << candidateMasterId_ << std::endl;
    // if masterId_ == 0, it means the UE was not attached to any eNodeB, so it only has to perform attachment procedures
    // if candidateMasterId_ == 0, it means the UE is detaching from its eNodeB, so it only has to perform detachment procedures

    if (masterId_ != 0)
    {
        // Delete Old Buffers
        deleteOldBuffers(masterId_);

        // amc calls
        LteAmc *oldAmc = getAmcModule(masterId_);
        oldAmc->detachUser(nodeId_, UL);
        oldAmc->detachUser(nodeId_, DL);
    }

    if (candidateMasterId_ != 0)
    {
        LteAmc *newAmc = getAmcModule(candidateMasterId_);
        assert(newAmc != nullptr);
        newAmc->attachUser(nodeId_, UL);
        newAmc->attachUser(nodeId_, DL);
    }

    // binder calls
    if (masterId_ != 0)
        binder_->unregisterNextHop(masterId_, nodeId_);

    if (candidateMasterId_ != 0)
    {
        binder_->registerNextHop(candidateMasterId_, nodeId_);
        das_->setMasterRuSet(candidateMasterId_);
    }
    binder_->updateUeInfoCellId(nodeId_,candidateMasterId_);

    // @author Alessandro Noferi
    if(getParentModule()->getParentModule()->findSubmodule("ueCollector") != -1)
    {
        binder_->moveUeCollector(nodeId_, masterId_, candidateMasterId_);
    }

    // change masterId and notify handover to the MAC layer
    MacNodeId oldMaster = masterId_;
  //  std::cout << "(doHandover) -- master id:" << masterId_ << " candidate master id: "<<candidateMasterId_<< std::endl;
    masterId_ = candidateMasterId_;

    mac_->doHandover(candidateMasterId_);  // do MAC operations for handover
    currentMasterRssi_ = candidateMasterRssi_;
    hysteresisTh_ = updateHysteresisTh(currentMasterRssi_);
    //std::cout << "(doHandover) -- master id:" << masterId_ << " candidate master id: "<<candidateMasterId_<< std::endl;


    // update cellInfo
    if (masterId_ != 0)
        cellInfo_->detachUser(nodeId_);

    if (candidateMasterId_ != 0)
    {
        CellInfo* oldCellInfo = cellInfo_;
        LteMacEnb* newMacEnb =  check_and_cast<LteMacEnb*>(getSimulation()->getModule(binder_->getOmnetId(candidateMasterId_))->getSubmodule("cellularNic")->getSubmodule("mac"));
        CellInfo* newCellInfo = newMacEnb->getCellInfo();
        newCellInfo->attachUser(nodeId_);
        cellInfo_ = newCellInfo;
        if (oldCellInfo == NULL)
        {
            // first time the UE is attached to someone
            int index = intuniform(0, binder_->phyPisaData.maxChannel() - 1);
            cellInfo_->lambdaInit(nodeId_, index);
            cellInfo_->channelUpdate(nodeId_, intuniform(1, binder_->phyPisaData.maxChannel2()));
        }
    }

    // update DL feedback generator
    LteDlFeedbackGenerator* fbGen;
    fbGen = check_and_cast<LteDlFeedbackGenerator*>(getParentModule()->getSubmodule("dlFbGen"));
    fbGen->handleHandover(masterId_);

    // collect stat
    emit(servingCell_, (long)masterId_);

    if (masterId_ == 0) {
        // EV << NOW << " LtePhyUe::doHandover - UE " << nodeId_ << " detached from the network" << endl;
//        std::cout << NOW << " LtePhyUe::doHandover - UE " << nodeId_ << " detached from the network" << endl;
    }
    else {
        // EV << NOW << " LtePhyUe::doHandover - UE " << nodeId_ << " has completed handover to gNB " << masterId_ << "... " << endl;
//        std::cout << NOW << " LtePhyUe::doHandover - UE " << nodeId_ << " has completed handover to gNB " << masterId_ << "... " << endl;
    }
        binder_->removeUeHandoverTriggered(nodeId_);

    // inform the UE's IP2Nic module to forward held packets
    IP2Nic* ip2nic =  check_and_cast<IP2Nic*>(getParentModule()->getSubmodule("ip2nic"));
    ip2nic->signalHandoverCompleteUe();

    // inform the eNB's IP2Nic module to forward data to the target eNB
    if (oldMaster != 0 && candidateMasterId_ != 0)
    {
        IP2Nic* enbIp2Nic =  check_and_cast<IP2Nic*>(getSimulation()->getModule(binder_->getOmnetId(masterId_))->getSubmodule("cellularNic")->getSubmodule("ip2nic"));
        enbIp2Nic->signalHandoverCompleteTarget(nodeId_,oldMaster);
    }

   // std::cout << "doHandover method -> handover done! new master id: " << masterId_ << " new candidate id: " << candidateMasterId_ << std::endl;
}

// TODO: ***reorganize*** method
void LtePhyUe::handleAirFrame(cMessage* msg)
{

    UserControlInfo* lteInfo = dynamic_cast<UserControlInfo*>(msg->removeControlInfo());
   // Ritika std::cout << "SHajib::LtePhyUe::handleAirFrame:SHAJIBSourceID " << lteInfo->getSourceId() << " Dest id: " << lteInfo->getDestId() << std::endl;
    if (useBattery_)
    {
        //TODO BatteryAccess::drawCurrent(rxAmount_, 0);
    }
    connectedNodeId_ = masterId_;
    LteAirFrame* frame = check_and_cast<LteAirFrame*>(msg);

    int sourceId = binder_->getOmnetId(lteInfo->getSourceId());
    if(sourceId == 0 )
    {
        // source has left the simulation
        delete msg;
        return;
    }

    // check if the air frame was sent on a correct carrier frequency
    double carrierFreq = lteInfo->getCarrierFrequency();
    LteChannelModel* channelModel = getChannelModel(carrierFreq);
    if (channelModel == NULL)
    {
        delete lteInfo;
        delete frame;
        return;
    }

    //Update coordinates of this user
    if (lteInfo->getFrameType() == HANDOVERPKT)
    {
        // check if the message is on another carrier frequency or handover is already in process
        if (carrierFreq != primaryChannelModel_->getCarrierFrequency() || (handoverTrigger_ != nullptr && handoverTrigger_->isScheduled()))
        {
            delete lteInfo;
            delete frame;
            return;
        }
     //   std::cout << "LtePhyUe::handleAirFrame) flag 1 - calling handoverHandler" << endl;
        handoverHandler(frame, lteInfo);
        return;
    }

    // Check if the frame is for us ( MacNodeId matches )
    if (lteInfo->getDestId() != nodeId_)
    {
        delete lteInfo;
        delete frame;
        return;
    }

        /*
         * This could happen if the ue associates with a new master while the old one
         * already scheduled a packet for him: the packet is in the air while the ue changes master.
         * Event timing:      TTI x: packet scheduled and sent for UE (tx time = 1ms)
         *                     TTI x+0.1: ue changes master
         *                     TTI x+1: packet from old master arrives at ue
         */
    if (lteInfo->getSourceId() != masterId_)
    {
        delete frame;
        return;
    }

        // send H-ARQ feedback up
    if (lteInfo->getFrameType() == HARQPKT || lteInfo->getFrameType() == GRANTPKT || lteInfo->getFrameType() == RACPKT)
    {
        handleControlMsg(frame, lteInfo);
        return;
    }
    if ((lteInfo->getUserTxParams()) != nullptr)
    {
        int cw = lteInfo->getCw();
        if (lteInfo->getUserTxParams()->readCqiVector().size() == 1)
            cw = 0;
        double cqi = lteInfo->getUserTxParams()->readCqiVector()[cw];
        emit(averageCqiDl_, cqi);
        recordCqi(cqi, DL);
    }
    // apply decider to received packet
    bool result = true;
    RemoteSet r = lteInfo->getUserTxParams()->readAntennaSet();
    if (r.size() > 1)
    {
        // DAS
        for (RemoteSet::iterator it = r.begin(); it != r.end(); it++)
        {
            /*
             * On UE set the sender position
             * and tx power to the sender das antenna
             */

//            cc->updateHostPosition(myHostRef,das_->getAntennaCoord(*it));
            // Set position of sender
//            Move m;
//            m.setStart(das_->getAntennaCoord(*it));
            RemoteUnitPhyData data;
            data.txPower=lteInfo->getTxPower();
            data.m=getRadioPosition();
            frame->addRemoteUnitPhyDataVector(data);
        }
        // apply analog models For DAS
        result=channelModel->isErrorDas(frame,lteInfo);
    }
    else
    {
        result = channelModel->isError(frame,lteInfo);
    }

    // update statistics
    if (result)
        numAirFrameReceived_++;
    else
        numAirFrameNotReceived_++;

    // EV << "Handled NRAirframe with ID " << frame->getId() << " with result "
//       << ( result ? "RECEIVED" : "NOT RECEIVED" ) << endl;

//    std::cout << "Handled NRAirframe with ID " << frame->getId() << " with result "
//           << ( result ? "RECEIVED" : "NOT RECEIVED" ) << endl;

    auto pkt = check_and_cast<inet::Packet *>(frame->decapsulate());

    // here frame has to be destroyed since it is no more useful
    delete frame;

    // attach the decider result to the packet as control info
    lteInfo->setDeciderResult(result);
    *(pkt->addTagIfAbsent<UserControlInfo>()) = *lteInfo;
    delete lteInfo;

    // send decapsulated message along with result control info to upperGateOut_
    send(pkt, upperGateOut_);

    if (getEnvir()->isGUI())
    updateDisplayString();
}

void LtePhyUe::handleUpperMessage(cMessage* msg)
{
//    if (useBattery_) {
//    TODO     BatteryAccess::drawCurrent(txAmount_, 1);
//    }

    auto pkt = check_and_cast<inet::Packet *>(msg);
    auto lteInfo = pkt->getTag<UserControlInfo>();

    MacNodeId dest = lteInfo->getDestId();
    if (dest != masterId_)
    {
        // UE is not sending to its master!!
        throw cRuntimeError("LtePhyUe::handleUpperMessage  UE preparing to send message to %d instead of its master (%d)", dest, masterId_);
    }

    double carrierFreq = lteInfo->getCarrierFrequency();
    LteChannelModel* channelModel = getChannelModel(carrierFreq);
    if (channelModel == NULL)
        throw cRuntimeError("LtePhyUe::handleUpperMessage - Carrier frequency [%f] not supported by any channel model", carrierFreq);


    if (lteInfo->getFrameType() == DATAPKT && (channelModel->isUplinkInterferenceEnabled() || channelModel->isD2DInterferenceEnabled()))
    {
        // Store the RBs used for data transmission to the binder (for UL interference computation)
        RbMap rbMap = lteInfo->getGrantedBlocks();
        Remote antenna = MACRO;  // TODO fix for multi-antenna
        binder_->storeUlTransmissionMap(channelModel->getCarrierFrequency(), antenna, rbMap, nodeId_, mac_->getMacCellId(), this, UL);
    }

    if (lteInfo->getFrameType() == DATAPKT && lteInfo->getUserTxParams() != nullptr)
    {
        double cqi = lteInfo->getUserTxParams()->readCqiVector()[lteInfo->getCw()];
        if (lteInfo->getDirection() == UL)
        {
            emit(averageCqiUl_, cqi);
            recordCqi(cqi, UL);
        }
        else if (lteInfo->getDirection() == D2D)
            emit(averageCqiD2D_, cqi);
    }

    LtePhyBase::handleUpperMessage(msg);
}
void LtePhyUe:: addToVC(double vehicleID, int curtowerID, std::vector<int> closestTowerLst, NazaninHandoverDecision::SpeedCategory speedCategory,double scalar, double predictedLSTM, double predictedTGNN){ //Ritika

//    if (oldMasterId_ != masterId_){
//        VCid.clear();
//    }
    if (simTime().dbl() != vc_cur_simtime)
        {
            VCid.clear();
            vc_cur_simtime = simTime().dbl();
        }
        if (simTime().dbl() == vc_cur_simtime)
           {

            if(scalar > predictedTGNN){   //Ritika
                //check if current tower and vehicle pair exists in the list. First check Vehicle
               // if(!checkIfCellTowerPairExistsInMap(curtowerID, vehicleID)){
                    // Add Current tower In the VC list
                    VCid.insert(std::make_pair(curtowerID, vehicleID));
                   // std::cout << "Shajib::Scalar:Current Tower  "<< curtowerID << " Added For Vehicle "<< vehicleID <<" AT:" <<simTime().dbl() << "Current Tower:  "<< curtowerID<< endl;

                    totalVCSize++;
                    virtualcellCountBySpeedLst[speedCategory]++;
               // }
            }
               for( auto towerId:closestTowerLst  ){

                  // if(!checkIfCellTowerPairExistsInMap(curtowerID, towerId)){
                      // std::cout << "Shajib:: Virtual Cell Created Tower Found: " << towerId <<  " Vehicle ID " << vehicleID <<  "  simTime().dbl(): "<< simTime().dbl() << endl;
                       VCid.insert(std::make_pair(towerId, vehicleID));
                       totalVCSize++;
                       testTotalClosestTowerFound++;
                       virtualcellCountBySpeedLst[speedCategory]++;

                //}
               }
           }
//        if(VCid.size() > 0){
//            std::cout<< "All Virtual Cell For Vehicle: " << vehicleID << endl;
//        }
//
//        for (VCid_itr = VCid.begin(); VCid_itr != VCid.end(); ++VCid_itr)
//                       {
//                          std::cout << "Shajib:: SimTime: " << simTime() << "  simTime().dbl(): "<< simTime().dbl() << " Vehicle: " << VCid_itr->first << " Tower(s) in VC: " << VCid_itr->second << "\nVC Size: " << currentVCSize << endl;
//                       }
}
bool LtePhyUe::checkIfCellTowerPairExistsInMap(int tower, double vehicle)
{
        auto itr1 = VCid.lower_bound(tower);
        auto itr2 = VCid.upper_bound(tower);

        while (itr1 != itr2)
        {
            if (itr1 -> first == tower && itr1 -> second == vehicle)
            {
              //  std::cout<< "Shajib:: checkIfCellTowerPairExistsInMap Vehicle " << vehicle << " TowerID : " << tower << " Found" << endl;
                return true;
            }
            itr1++;
        }
        return false;
}
bool LtePhyUe:: checkIfTowerExistsInMap(int towerId){

    auto pos = VCid.find(towerId);
          if (towerId == pos->first){
              return true;
          }
    return false;
}

double LtePhyUe::updateHysteresisTh(double v)   // v is same as x
{
    if (hysteresisFactor_ == 0)
        return 0;
    else
        return ((v / hysteresisFactor_) - 5);
}

double LtePhyUe::updateHysteresisThMinSinr(double v)   // v is same as x
{
    if (hysteresisFactorSinr_ == 0)
        return 0;
    else
        return ((v / hysteresisFactorSinr_) - 3);
}

double LtePhyUe::updateHysteresisThMinRsrp(double v)
{
    if (hysteresisFactorRsrp_ == 0)
        return 0;
    else
        return ((v / hysteresisFactorRsrp_) + 20);
}

double LtePhyUe::updateHysteresisThMaxDist(double v)
{
    if (hysteresisFactorDist_ == 0)
        return 0;
    else
        return ((v / hysteresisFactorDist_) + 150) ;
}

double LtePhyUe::updateHysteresisTowerLoad(double v)
{
    if (hysteresisFactorLoad_ == 0)
        return 0;
    else
        return ((v / hysteresisFactorLoad_) + 0.5) ;
}

void LtePhyUe::deleteOldBuffers(MacNodeId masterId)
{
    /* Delete Mac Buffers */

    // delete macBuffer[nodeId_] at old master
    LteMacEnb *masterMac = check_and_cast<LteMacEnb *>(getMacByMacNodeId(masterId));
    masterMac->deleteQueues(nodeId_);

    // delete queues for master at this ue
    mac_->deleteQueues(masterId_);

    /* Delete Rlc UM Buffers */

    // delete UmTxQueue[nodeId_] at old master
    LteRlcUm *masterRlcUm = check_and_cast<LteRlcUm*>(getRlcByMacNodeId(masterId, UM));
    masterRlcUm->deleteQueues(nodeId_);

    // delete queues for master at this ue
    rlcUm_->deleteQueues(nodeId_);

    /* Delete PDCP Entities */
    // delete pdcpEntities[nodeId_] at old master
    LtePdcpRrcEnb* masterPdcp = check_and_cast<LtePdcpRrcEnb *>(getPdcpByMacNodeId(masterId));
    masterPdcp->deleteEntities(nodeId_);

    // delete queues for master at this ue
    pdcp_->deleteEntities(masterId_);
}

DasFilter* LtePhyUe::getDasFilter()
{
    return das_;
}

void LtePhyUe::sendFeedback(LteFeedbackDoubleVector fbDl, LteFeedbackDoubleVector fbUl, FeedbackRequest req)
{
    Enter_Method("SendFeedback");
    // EV << "LtePhyUe: feedback from Feedback Generator" << endl;

    //Create a feedback packet
    auto fbPkt = makeShared<LteFeedbackPkt>();
    //Set the feedback
    fbPkt->setLteFeedbackDoubleVectorDl(fbDl);
    fbPkt->setLteFeedbackDoubleVectorDl(fbUl);
    fbPkt->setSourceNodeId(nodeId_);

    auto pkt = new Packet("feedback_pkt");
    pkt->insertAtFront(fbPkt);

    UserControlInfo* uinfo = new UserControlInfo();
    uinfo->setSourceId(nodeId_);
    uinfo->setDestId(masterId_);
    uinfo->setFrameType(FEEDBACKPKT);
    uinfo->setIsCorruptible(false);
    // create LteAirFrame and encapsulate a feedback packet
    LteAirFrame* frame = new LteAirFrame("feedback_pkt");
    frame->encapsulate(check_and_cast<cPacket*>(pkt));
    uinfo->feedbackReq = req;
    uinfo->setDirection(UL);
    simtime_t signalLength = TTI;
    uinfo->setTxPower(txPower_);
    // initialize frame fields

    frame->setSchedulingPriority(airFramePriority_);
    frame->setDuration(signalLength);

    uinfo->setCoord(getRadioPosition());

    //TODO access speed data Update channel index
//    if (coherenceTime(move.getSpeed())<(NOW-lastFeedback_)){
//        cellInfo_->channelIncrease(nodeId_);
//        cellInfo_->lambdaIncrease(nodeId_,1);
//    }
    lastFeedback_ = NOW;

    // send one feedback packet for each carrier
    std::map<double, LteChannelModel*>::iterator cit = channelModel_.begin();
    for (; cit != channelModel_.end(); ++cit)
    {
        double carrierFrequency = cit->first;
        LteAirFrame* carrierFrame = frame->dup();
        UserControlInfo* carrierInfo = uinfo->dup();
        carrierInfo->setCarrierFrequency(carrierFrequency);
        carrierFrame->setControlInfo(carrierInfo);

        // EV << "LtePhy: " << nodeTypeToA(nodeType_) << " with id "
//           << nodeId_ << " sending feedback to the air channel for carrier " << carrierFrequency << endl;
        sendUnicast(carrierFrame);
    }

    delete frame;
    delete uinfo;
}

void LtePhyUe::recordCqi(unsigned int sample, Direction dir)
{
    if (dir == DL)
    {
        cqiDlSamples_.push_back(sample);
        cqiDlSum_ += sample;
        cqiDlCount_++;
    }
    if (dir == UL)
    {
        cqiUlSamples_.push_back(sample);
        cqiUlSum_ += sample;
        cqiUlCount_++;
    }
}

double LtePhyUe::getAverageCqi(Direction dir)
{
    if (dir == DL)
    {
        if (cqiDlCount_ == 0)
            return 0;
        return (double)cqiDlSum_/cqiDlCount_;
    }
    if (dir == UL)
    {
        if (cqiUlCount_ == 0)
            return 0;
        return (double)cqiUlSum_/cqiUlCount_;
    }
}

double LtePhyUe::getVarianceCqi(Direction dir)
{
    double avgCqi = getAverageCqi(dir);
    double err, sum = 0;

    if (dir == DL)
    {
        for (auto it = cqiDlSamples_.begin(); it != cqiDlSamples_.end(); ++it)
        {
            err = avgCqi - *it;
            sum = (err * err);
        }
        return sum/cqiDlSamples_.size();
    }
    if (dir == UL)
    {
        for (auto it = cqiUlSamples_.begin(); it != cqiUlSamples_.end(); ++it)
        {
            err = avgCqi - *it;
            sum = (err * err);
        }
        return sum/cqiUlSamples_.size();
    }
}

void LtePhyUe::finish()
{
    if (getSimulation()->getSimulationStage() != CTX_FINISH)
    {
        // do this only if this PHY layer is connected to a serving base station
        if (masterId_ > 0)
        {
            // clear buffers
            deleteOldBuffers(masterId_);

            // amc calls
            LteAmc *amc = getAmcModule(masterId_);
            if (amc != nullptr)
            {
                amc->detachUser(nodeId_, UL);
                amc->detachUser(nodeId_, DL);
            }

            // binder call
            binder_->unregisterNextHop(masterId_, nodeId_);

            // cellInfo call
            cellInfo_->detachUser(nodeId_);
        }
//
//        for (VCid_itr = VCid.begin(); VCid_itr != VCid.end(); ++VCid_itr)
//                    {
//                       // std::cout << "Shajib:: SimTime: " << simTime() << "  simTime().dbl(): "<< simTime().dbl() << " Vehicle: " << VCid_itr->first << " Tower(s) in VC: " << VCid_itr->second << "\nVC Size: " << currentVCSize << endl;
//                    }
    }
}

///////////////////////////////////////////////////




void LtePhyUe::performanceAnalysis()
{
    std::array<std::string, NazaninHandoverDecision::SPEED_COUNT> speedNames = {
           "SPEED_020", "SPEED_2140", "SPEED_4160", "SPEED_6180",
           "SPEED_81100", "SPEED_101120", "SPEED_121140", "SPEED_141160", "SPEED_160PLUS"
       };

    // Ritika start --- File to write results
        std::ofstream outFile("/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/performanceAnalysis.txt", std::ios::app);
        if (!outFile.is_open()) {
            std::cerr << "Error opening file for writing!" << std::endl;
            return;
        } //Ritika end

    if (simTime().dbl() == 100 && !isPerformedAnalysis )
    {
        isPerformedAnalysis = true; // only one time is needed
        //  std::cout << "Simtime - " << simTime().dbl() << std::endl;

        //Ritika start
        outFile << "testHODecisionByScalarCount: " << testHODecisionByScalarCount
                        << " testHODecisionBySpeedCount: " << testHODecisionBySpeedCount
                        << " totalScalarParaConditionedPassed: " << totalScalarParaConditionedPassed
                        << " testMasterIDCount: " << testMasterIDCount
                        << " testTotalClosestTowerFound: " << testTotalClosestTowerFound
                        << "Doing Ritika New TGNN Testing"  << "\n\n";

        outFile << "Total Vehicle: " << vehicleCountTotal << " Simtime: " << simTime().dbl() << "\n\n";

        // INTRA VC
        outFile << "---------------------------------------------INTRA-----------------------------------------------------\n\n";
        outFile << "INTRA VC HO Total: " << insideVCHOTotal << " Avg Num of Intra-VC HO: " << insideVCHOTotal / vehicleCountTotal << "\n\n";
        outFile << "INTRA VC: Cumulative Time: " << insideVCHOTime << ", Avg INTRA VC HO Time: " << insideVCHOTime / insideVCHOTotal << "\n\n";

        outFile << "-------------------Percentages of INTRA-VC HO vs speed---------\n";
        for (int speed = 0; speed < NazaninHandoverDecision::SPEED_COUNT; ++speed) {
                    outFile << "Speed category: " << speedNames[speed]
                            << " Total: " << insideHOVehicleLst[speed]
                            << " Percentage: " << (insideHOVehicleLst[speed] / insideVCHOTotal) << "\n";
         }

         // INTER VC
         outFile << "\n\n--------------------------------------------INTER/OUTSIDE VC----------------------------------------------\n\n";
         outFile << "INTER VC HO Total: " << outsideVCHOTotal << " Avg Num of Inter-VC HO: " << outsideVCHOTotal / vehicleCountTotal << "\n\n";
         outFile << "INTER VC: Cumulative Time: " << outsideVCHOTime << ", Avg INTER VC HO Time: " << outsideVCHOTime / outsideVCHOTotal << "\n\n";

         outFile << "-------------------Percentages of INTER-VC HO vs speed---------\n";
         for (int speed = 0; speed < NazaninHandoverDecision::SPEED_COUNT; ++speed) {
                    outFile << "Speed Category: " << speedNames[speed]
                            << " Total: " << outsideHOVehicleLst[speed]
                            << " Percentage: " << (outsideHOVehicleLst[speed] / outsideVCHOTotal) << "\n";
          }

          // Failed HO
          outFile << "\n\n----------------------------------------------Failed HO-----------------------------------------------------\n";
          outFile << "Failed VC HO Total: " << failureHOTotal << " Avg Num of Failed HO: " << failureHOTotal / vehicleCountTotal << "\n\n";

          // Ping Pong HO
          outFile << "-------------------------------------------------Ping Pong---------------------------------------------------\n";
          outFile << "Ping Pong HO Total: " << pingPongHOTotal << ", Avg Num of Ping Pong HO: " << pingPongHOTotal / vehicleCountTotal << "\n\n";

          // VC Size
          outFile << "Total Virtual Cell: " << totalVCSize << " VC Size Ratio: " << (totalVCSize / vehicleCountTotal) / 100 << "\n\n";

          outFile << "----------------------------------------------Size of VC over speed-----------------------------------------\n\n";
          for (int speed = 0; speed < NazaninHandoverDecision::SPEED_COUNT; ++speed) {
                    outFile << "Virtual Cell Size For Speed: " << speedNames[speed]
                            << " VC Size: " << virtualcellCountBySpeedLst[speed]
                            << " Total Vehicle: " << vehicleCountBySpeedLst[speed]
                            << " Average Size: " << virtualcellCountBySpeedLst[speed] / vehicleCountBySpeedLst[speed] << "\n\n";
          } //Ritika end

        /*  Commenting out Shajib's print out to console
        std::cout << " testHODecisionByScalarCount: " << testHODecisionByScalarCount << "testHODecisionBySpeedCount: " << testHODecisionBySpeedCount  << " totalScalarParaConditionedPassed: "<< totalScalarParaConditionedPassed<< " testMasterIDCount: " << testMasterIDCount << " testTotalClosestTowerFound: "<< testTotalClosestTowerFound << endl;

        // Vehicle Count
        std::cout << "  Total Vehicle  " << vehicleCountTotal << " Simtime "<< simTime().dbl()<<" \n\n"<< std::endl;


        // ------------------------------------------------------- INTRA -----------------------------------------------------
        std::cout<< "   ---------------------------------------------INTRA-----------------------------------------------------\n\n" <<endl;
        std::cout << "  INTRA VC HO Total: " << insideVCHOTotal << " Avg Num of Intra-VC HO:"<< insideVCHOTotal/vehicleCountTotal <<"\n\n"<< std::endl;
        std::cout << "  INTRA VC: Cumulitive Time: " << insideVCHOTime << ", Avg INTRA VC HO Time: " << insideVCHOTime / insideVCHOTotal <<" \n\n"<< std::endl;
        std::cout<< "-------------------Percentages of INTRA-VC HO vs speed---------\n" <<endl;

        for (int speed = 0; speed < NazaninHandoverDecision::SPEED_COUNT; ++speed) {
             // Speed Dependent INTRA VC HO
             std::cout << " Speed category:  " << speedNames[speed] << " Total: " << insideHOVehicleLst[speed] << "  Percentage:  " << (insideHOVehicleLst[speed]/insideVCHOTotal) << std::endl;
             // Speed Dependent HO Ping Pong
            // std::cout << "Ping Pong HO fro Speed " << speedNames[speed] << " : " << pingPongHOVehicleLst[speed] << "\n\n"<< std::endl;
         }

         ///------------------------------------------ I--------------INTER--------------------------------------------------------
        std::cout<< "\n\n--------------------------------------------INTER/OUTSIDE VC----------------------------------------------\n\n" <<endl;
        std::cout << "  INTER VC HO Total: " << outsideVCHOTotal << " Avg Num of Inter-VC HO:"<< outsideVCHOTotal/vehicleCountTotal <<"\n\n"<< std::endl;
        std::cout << "  INTER VC: Cumulitive Time: " << outsideVCHOTime << ", Avg INER VC HO Time: " << outsideVCHOTime / outsideVCHOTotal <<" \n\n"<< endl;

        std::cout<< "-------------------Percentages of INTER-VC HO vs speed---------\n" <<endl;
        for (int speed = 0; speed < NazaninHandoverDecision::SPEED_COUNT; ++speed) {
                // Speed Dependent INTER VC HO
                std::cout << "Speed Category: " << speedNames[speed] << " Total: " << outsideHOVehicleLst[speed] << "  Percentage:  " << (outsideHOVehicleLst[speed]/outsideVCHOTotal)     << std::endl;

         }

        ///-------------------------------------------------------- Failed HO--------------------------------------------------------
        std::cout<< "\n\n----------------------------------------------Failed HO-----------------------------------------------------\n" <<endl;
        std::cout << "  Failed VC HO Total: " << failureHOTotal << " Avg Num of Failed HO:"<< failureHOTotal/vehicleCountTotal <<"\n\n"<< std::endl;
       // std::cout << "Failed VC: Cum HO Time: " <<  << ", Avg INER VC HO Time: " << outsideVCHOTime / failureHOTotal <<" \n\n"<< endl;

        //-------------------------------------------------------------HO Ping Pong--------------------------------------------------
        std::cout<< "   -------------------------------------------------Ping Pong---------------------------------------------------\n" <<endl;
         std::cout << " Ping Pong HO Total: " << pingPongHOTotal << ", Avg Num of Ping Pong HO: " << pingPongHOTotal / vehicleCountTotal <<" \n\n"<< std::endl;


         //-------------------------------------------------------VC SIZE---------------------------------------------------------------

        //-----------------------------------------------Number OF Virtual Cell Size By Speed Category --------------------------------------------------------
        std::cout << "  Total virtual Cell: " <<  totalVCSize << " VC Size Ratio  (totalVCSize / vehicleCountTotal)/100: "<< (totalVCSize / vehicleCountTotal)/100 <<" \n\n"<< endl;

        std::cout << "----------------------------------------------Size of VC over speed-----------------------------------------\n\n" << endl;
        for (int speed = 0; speed < NazaninHandoverDecision::SPEED_COUNT; ++speed) {
                  // Speed Dependent INTRA VC HO
                  std::cout << "    Virtual Cell Size For Speed:  " << speedNames[speed] << "=> : VC Size: "  << virtualcellCountBySpeedLst[speed] << " Total Vehicle: "<<  vehicleCountBySpeedLst[speed] << "  Average Size:(virtualcellCountBySpeedLst[speed] / vehicleCountBySpeedLst[speed]):  "<< virtualcellCountBySpeedLst[speed] / vehicleCountBySpeedLst[speed] <<"\n"<< std::endl;
        }

        */ //ending commenting out Shajib


        finishSimTime = simTime().dbl();
    }

    // Ritika start
    // Save result at the end of simulation with timestamp
    if (simTime().dbl() != finishSimTime && simTime().dbl() > 100)
    {
            time_t t = time(0);
            struct tm* now = localtime(&t);
            char buffer[80];
            strftime(buffer, 80, "%Y%m%d%H%M%S", now);

            std::string oldFile = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/performanceAnalysis.txt";
            std::string newFile = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/performanceAnalysis_" + std::string(buffer) + ".txt";
            rename(oldFile.c_str(), newFile.c_str());
     }
    outFile.close();//Ritika end

    /* Commenting out Shajib

    // saving result at the end of simulation in a file
    if (simTime().dbl() != finishSimTime && simTime().dbl() > 100)
    {
        int result;
        time_t t = time(0);   // get time now
        struct tm *now = localtime(&t);

        char buffer[80];
        strftime(buffer, 80, "%Y%m%d%H%M%S", now);

        //char oldname[] = "/home/shajib/Simulation/Myversion2/WorkFolder/BatchRunResult/Test/BatchExecution.txt";
        char oldname[] = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/BatchRunResult/Test/BatchExecution.txt";
        std::string str(buffer);
        //std::string strNewFile = "/home/shajib/Simulation/Myversion2/ConsoleOutput/" + str + ".txt";
        std::string strNewFile = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/Simulation/Myversion2/ConsoleOutput/" + str + ".txt";
        int len = strNewFile.length();
        char buffer_new[len + 1];
        strcpy(buffer_new, strNewFile.c_str());
        result = rename(oldname, buffer_new);
    }
     */ //ending commenting out Shajib
}

void LtePhyUe::handlenormalHandover(double rssi, double rsrq, double maxSINR, double maxRSRP, double speedDouble, double distanceDouble, NazaninHandoverDecision::SpeedCategory speedCategory, bool isIntraVCHO) {



    candidateMasterId_ = sel_srv_Qvalue_id;
    oldMasterId_ = candidateMasterId_;
    candidateMasterRssi_ = rssi;
    candidateMasterSinr_ = maxSINR;
    candidateMasterRsrp_ = maxRSRP;
    candidateMasterDist_ = distanceDouble;
    candidateMasterSpeed_ = speedDouble;

    hysteresisTh_ = updateHysteresisTh(rssi);

    binder_->addHandoverTriggered(nodeId_, masterId_, candidateMasterId_);

    tLoad[candidateMasterId_]++;
    tLoad[masterId_]--;


    ho_Qvalue = rsrq + (200 - maxSINR);
    ho_rssi = rssi;
    ho_sinr = maxSINR;
    ho_rsrp = maxRSRP;
    ho_dist = distanceDouble;
    ho_load = avgLoad;
    avg_srv_QvalueV.clear();

    // If we find the candidate master id in a vector of master ids that is updated every 2 simtimes,
    // it means we have had this tower HO before and we have a ping pong situation
    // incrementing number of ping pong HOs
    if (std::find(last_srv_MasterIdV.begin(), last_srv_MasterIdV.end(), candidateMasterId_) != last_srv_MasterIdV.end()) {
        pingPongHOTotal++;
        pingPongHOVehicleLst[speedCategory]++;
        pingPongHOVehiclTotal++;
    }
    else if(isIntraVCHO){

        insideVCHOTotal++;
        insideVCHOVehicleTotal++;
        insideVCHOTime = comHOTime;
        insideHOVehicleLst[speedCategory]++;
    }
    else{
        // Inter HO / Outside VC HO
        outsideVCHOTotal++;
        outsideVCHOVehicleTotal++;
        outsideVCHOTime = comHOTime + handoverLatency_ + handoverDetachment_ + handoverAttachment_ + handoverDelta_ + hoVC;
        outsideHOVehicleLst[speedCategory]++;
    }
    if (!handoverStarter_->isScheduled()) {
        handoverStarter_ = new cMessage("handoverStarter");
        scheduleAt(simTime() + handoverDelta_, handoverStarter_);
    }

    //////////Ritika TGNN diff code started

    std::ofstream tgnnFile;
    tgnnFile.open(
        "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/"
        "simu5G/src/stack/phy/layer/inputTGNNdiff.txt",
        std::ios::app
    );

    tgnnFile
        << simTime().dbl() << " "
        << getMacNodeId() << " "
        << rssi << " "
        << maxSINR << " "
        << speedDouble << " "
        << distanceDouble
        << std::endl;

    tgnnFile.close();

    //////////Ritika tgnn diff code ended


}

void LtePhyUe::handleFailureHandover(MacNodeId candidateID, double rssi, double rsrq, double maxSINR, double maxRSRP, double speedDouble, double distanceDouble,NazaninHandoverDecision::SpeedCategory speedCategory) {


    //std::cout<< "Shajib:: Trying handleFailureHandover Handover: candidateID " << candidateID << " oldMasterId_ " << oldMasterId_ << " Master ID " << masterId_ << endl;
    candidateMasterId_ = candidateID;
    oldMasterId_ = candidateMasterId_;
    candidateMasterRssi_ = rssi;
    candidateMasterSinr_ = maxSINR;
    candidateMasterRsrp_ = maxRSRP;
    candidateMasterDist_ = distanceDouble;
    candidateMasterSpeed_ = speedDouble;
//
//
//    ho_rssi = rssi;
//    ho_sinr = maxSINR;
//    ho_rsrp = maxRSRP;
//    ho_dist = distanceDouble;

    failureHOTotal++;
    failureHOVehicleTotal++;
    failureHOTime = comHOTime + handoverLatency_ + handoverDetachment_ + handoverAttachment_ + handoverDelta_ + hoVC;
    failureHOVehicleLst[speedCategory]++;


    binder_->addHandoverTriggered(nodeId_, masterId_, candidateMasterId_);
    if (!handoverStarter_->isScheduled()) {
      //  std::cout << "method: handleFailureHandover -- scheduling a handover in : " << handoverDelta_ << " plus our simtime: " << simTime() << std::endl;
       // handoverStarter_ = new cMessage("handoverStarter");
        scheduleAt(simTime() + handoverDelta_, handoverStarter_);
    }
}

void LtePhyUe::updateCurrentMaster(double rssi, double maxSINR, double maxRSRP, double distanceDouble, double speedDouble) {
    currentMasterRssi_ = rssi;
    currentMasterSinr_ = maxSINR;
    currentMasterRsrp_ = maxRSRP;
    currentMasterDist_ = distanceDouble;
    currentMasterSpeed_ = speedDouble;

    candidateMasterRssi_ = rssi;
    candidateMasterSinr_ = maxSINR;
    candidateMasterRsrp_ = maxRSRP;
    candidateMasterDist_ = distanceDouble;
    candidateMasterSpeed_ = speedDouble;
}

void LtePhyUe::performHysteresisUpdate(double currentMasterRssi_, double currentMasterSinr_, double currentMasterRsrp_, double currentMasterDist_) {

    hysteresisTh_ = updateHysteresisTh(currentMasterRssi_);
    hysteresisSinrTh_ = updateHysteresisThMinSinr(currentMasterSinr_);
    hysteresisRsrpTh_ = updateHysteresisThMinRsrp(currentMasterRsrp_);
    hysteresisDistTh_ = updateHysteresisThMaxDist(currentMasterDist_);
    hysteresisLoadTh_ = updateHysteresisTowerLoad(avgLoad);

}
