//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
//



#include <assert.h>
#include "stack/phy/layer/NazaninHandoverDecision.h"
#include "stack/phy/layer/LtePhyUe.h"

//#include "common/LteControlInfo.h"
//#include "stack/phy/packet/LteAirFrame_m.h"
//#include "stack/phy/ChannelModel/LteChannelModel.h"

#include <ctime>
#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>
#include <cstdlib>
double bsLoadCal1Weighted;
const int NUM_TOWERS = 10;
double bsLoad[NUM_TOWERS] = {0};  // Use an array to store bsLoad values
double bsLoadCal1[NUM_TOWERS] = {0};
double minLoad = -10;
double minTowerLoad, avgLoad, sumbsLoadCal1, towerCount;
double towerLoad_cur_simtime = 1;

std::string baseFilePath = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/";
std::string speedFile = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/ChannelModel/speedFile.txt";


//std::array<double, SPEED_COUNT> vinsideHO;
//std::array<double, SPEED_COUNT> fho_vinsideHO;
//std::array<double, SPEED_COUNT> vehicleCount;
//std::array<double, SPEED_COUNT> insideHO;
//std::array<double, SPEED_COUNT> fho_insideHO;
//std::array<double, SPEED_COUNT> pho_insideHO;
//std::array<double, SPEED_COUNT> vvehicleCount;
//
//
//
//double countTotalHO, insideVCHO, fho_insideHOTotal, pho_insideHOTotal;
//
//double vehicleCountTotal = 0;
//
//double comHOTime = 0, insideVCHOTime = 0;
//double vSpeed;
//double updt_simtime = 1;
//double finishSimTime = 1;
//double lstmSimTime = 1;

//   /**
//    * Hysteresis threshold to evaluate handover: it introduces a small polarization to
//    * avoid multiple subsequent handovers
//    */
//   double hysteresisTh_; //HO threshold value
//   double hysteresisSinrTh_; //HO threshold value
//   double hysteresisRsrpTh_; //HO threshold value
//   double hysteresisDistTh_; //HO threshold value
//   double hysteresisSpeedTh_; //HO threshold value
//   double hysteresisLoadTh_; //HO threshold value
//
//   /**
//    * Value used to divide currentMasterRssi_ and create an hysteresisTh_
//    * Use zero to have hysteresisTh_ == 0.
//    */
//   // TODO: bring it to ned par!
//   double hysteresisFactor_;
//   double hysteresisFactorSinr_;
//   double hysteresisFactorRsrp_;
//   double hysteresisFactorDist_;
//   double hysteresisFactorLoad_;

LtePhyUe* lte = new LtePhyUe();

enum SpeedCategory {
    SPEED_020, SPEED_2140, SPEED_4160, SPEED_6180, SPEED_81100, SPEED_101120, SPEED_121140, SPEED_141160, SPEED_160PLUS, SPEED_COUNT
};

UserControlInfo *lteInfo = new UserControlInfo();

NazaninHandoverDecision::NazaninHandoverDecision() {
//    std::cout <<"PRINTING IN NAZANINHANDOVERMNG" <<std::endl;
    // TODO Auto-generated constructor stub

}

NazaninHandoverDecision::~NazaninHandoverDecision() {
    // TODO Auto-generated destructor stub
}

std::vector<int> NazaninHandoverDecision::GetClosestTowersId(double xCoordRef, double yCoordRef, int vehicleID, int curTower){

    std::vector<int> closestTower;
    VehiclePosCal.x = xCoordRef;
    VehiclePosCal.y= yCoordRef;
    VehiclePosCal.z = 0;



    for (auto tower : Tower_Position){

        towerPosition.x = tower.second.first;
        towerPosition.y= tower.second.second;
        towerPosition.z = 300; // 300 is fixed for every tower
        double towerDistance = towerPosition.distance(VehiclePosCal);

        if(towerDistance < 600){
           // std::cout <<  "Shajib::Found Closest Tower For Vehicle: " << vehicleID <<" CurTower: "<< curTower << " Close Tower:  " << tower.first << " Distance: " << towerDistance <<" xCoordRef: "<< xCoordRef << " yCoordRef " << yCoordRef << endl;

            closestTower.push_back(tower.first);
        }
        else{
            //std::cout <<  "Shajib::Not Found. Tower:" << tower.first << " Distance: " << towerDistance << " Vehicle " << vehicleID << " xCoordRef: "<< xCoordRef << " yCoordRef " << yCoordRef << endl;
        }

    }


return  closestTower;
}
//
std::tuple<double, double, double, double> NazaninHandoverDecision::calculateMetrics(LteChannelModel* primaryChannelModel_, LteAirFrame* frame, UserControlInfo* lteInfo) {
//    std::cout <<"calculateMetrics IN NazaninHandoverDecision" <<std::endl;
//    std::cout << ", Tower ID: " << lteInfo->getSourceId() << ", UE ID: " << lteInfo->getDestId() << std::endl;
//    std::cout << ", Tower ID: " << lteInfo->getSourceId() << ", UE ID: " << lteInfo->getDestId() << std::endl;
        double rssi = 0;
        std::vector<double> rssiV = primaryChannelModel_->getSINR(frame, lteInfo);
        std::vector<double>::iterator it;
        // Calculate average RSSI
        for (it = rssiV.begin(); it != rssiV.end(); ++it){
                    rssi += *it;
                }
        std::cout<<endl;

        rssi /= rssiV.size();

        double maxSINR = *max_element(rssiV.begin(), rssiV.end());
//        std::cout <<  "MAX SINR :" << maxSINR << std::endl;
         //calculate RSRP
        std::vector<double> rsspV = primaryChannelModel_->getRSRP(frame, lteInfo);
        double maxRSRP = *max_element(rsspV.begin(), rsspV.end());
//        std::cout << "RSRP :" << maxRSRP << std::endl;
        //calculate RSRQ
        double rsrq = (10 * maxRSRP) / rssi;

        return std::make_tuple(rssi, maxSINR, maxRSRP, rsrq);
    }

double NazaninHandoverDecision::getParfromFile(std::string filepath){
    std::ifstream file(filepath);
    std::string parData;
    double parDouble = 0;
    while (std::getline (file, parData))
    {
        parDouble = atof(parData.c_str());
    }
    file.close();
    return parDouble;
}
std::tuple<double, double> NazaninHandoverDecision::getParfromFileForSVR(std::string filepath){

    std::ifstream file(filepath);
    std::string line;
    double xCoord, yCoord;
    double data[2] = {0,0};
    int count = 0;
    while (std::getline (file, line))
    {
        std::istringstream iss(line);
        std::string token;
        while(std::getline(iss, token, ' '))
        {
            data[count] = atof(token.c_str());
            //std::cout<< "Shajib TUPE CHECK:  data[" << count << "]" << data[count]<<" " << endl;
            count++;
        }

    }
    file.close();
  //  std::cout<< "Shajib TUPE CHECK: " << data[0]<<" "<< data[1] << endl;
 return std::make_tuple(data[0],data[1]);
}

void NazaninHandoverDecision::calculateTowerLoad(UserControlInfo* lteInfo, LteAirFrame* frame) {
       int index = lteInfo->getSourceId()-1;
//       std::cout << "LtePhyUe::calculateTowerLoad -> Tower: " << index << endl;
       bsLoad[index]++;
//       std::cout << "LtePhyUe::calculateTowerLoad -> bsLoad: " << bsLoad[index] << endl;
       bsLoadCal1Weighted = (bsLoad[index] + 10) / (std::accumulate(bsLoad, bsLoad + NUM_TOWERS, 0) + 10);
//       std::cout << "LtePhyUe::calculateTowerLoad -> bsLoadCal1Weighted: " << bsLoadCal1Weighted <<std::endl;

       if (simTime().dbl() != towerLoad_cur_simtime)   //high value means better, less vehicle connected
          {
              minTowerLoad = towerLoad(frame, lteInfo);
              towerCount++;
          }
          sumbsLoadCal1 = bsLoadCal1Weighted + minTowerLoad;
          if((int)simTime().dbl() % 10 == 0)
          {
              avgLoad = sumbsLoadCal1 / 10;
              avgLoad = 1 - avgLoad;
              sumbsLoadCal1 = 0;
              towerCount = 0;
              towerLoad_cur_simtime = simTime().dbl();
//              std::cout << "Tower: " << lteInfo->getSourceId() <<" Avg Load: " << avgLoad << endl;
          }
//          std::cout << "LtePhyUe::calculateTowerLoad -> Tower: " << lteInfo->getSourceId() <<" Avg Load: " << avgLoad << endl;
//          std::cout << "LtePhyUe::calculateTowerLoad -> minTowerLoad: " << minTowerLoad << " tower count: "<< towerCount <<std::endl;



}

double NazaninHandoverDecision::towerLoad(LteAirFrame* frame, UserControlInfo* lteInfo) {
    int flag=0;
    for (int i = 0; i < NUM_TOWERS; i++) {
        bsLoadCal1[i] = (bsLoad[i] + 10) / (std::accumulate(bsLoad, bsLoad + NUM_TOWERS, 0) - bsLoad[i] + 10);
        if (bsLoadCal1[i] >= minLoad && flag!=1) {
            minLoad = bsLoadCal1[i];
            flag=1;
        }
    }
//    std::cout << "minLoad in towerLoad function: "<< minLoad << std::endl;

    std::fill(std::begin(bsLoad), std::end(bsLoad), 0);
    towerLoad_cur_simtime = simTime().dbl();
//    std::cout << "towerLoad_cur_simtime in towerLoad function: "<< towerLoad_cur_simtime << std::endl;

    return minLoad;
}
//
void NazaninHandoverDecision::saveParaToFile(std::string filepath, double para) {
    std::ofstream file(baseFilePath+filepath);
    file << para << "\t";
    file.close();
}
void NazaninHandoverDecision::saveStringParaToFile(std::string filepath, std::string para) {
    std::ofstream file(baseFilePath+filepath, std::ios_base::app);
    file << para << endl;
    file.close();
}

void NazaninHandoverDecision::saveArrayToFile(const std::string& fileName, const std::vector<double>& array) {
    std::ofstream file(baseFilePath + fileName);
    for (const auto& value : array) {
        file << value << "\t";
    }
}

void NazaninHandoverDecision::runLSTM() {
    std::string pypredLSTMFile;
    std::string pypredLSTM_CmdPyCpp = "python3 " + baseFilePath + "predLSTM.py";
    pypredLSTM_CmdPyCpp += pypredLSTMFile;
    system(pypredLSTM_CmdPyCpp.c_str());
}


////////////////////////Ritika///////////////////////////////
void NazaninHandoverDecision::runTGNN() {
    std::string pypredTGNNFile;
    std::string pypredTGNN_CmdPyCpp = "python3 " + baseFilePath + "infer_proper_tgnn.py";
    pypredTGNN_CmdPyCpp += pypredTGNNFile;
    system(pypredTGNN_CmdPyCpp.c_str());
}


//void NazaninHandoverDecision::runTGNNdiff() {
  //  std::string pypredTGNNdiffFile;
    //std::string pypredTGNNdiff_CmdPyCpp = "python3 " + baseFilePath + "predTGNNdiff.py";
    //pypredTGNNdiff_CmdPyCpp += pypredTGNNdiffFile;
    //system(pypredTGNNdiff_CmdPyCpp.c_str());
//}

void NazaninHandoverDecision::runTGNNdiff() {
    static bool properTGNNStarted = false;
    if (properTGNNStarted) {
        return; // don't spawn again
    }
    properTGNNStarted = true;

    // If you actually need args, build them into pypredTGNNdiffFile first
    //std::string pypredTGNNdiffFile; // currently empty in your code

    std::string cmd = "python3 " + baseFilePath + "infer_proper_tgnn.py";
    //cmd += pypredTGNNdiffFile;

    // IMPORTANT: run in background so OMNeT++ doesn't block
    cmd += " > /tmp/properTGNN.log 2>&1 &";

    int ret = std::system(cmd.c_str());
    EV_INFO << "[ProperTGNN] Started infer_proper_tgnn.py async, system() returned " << ret << "\n";
}

void NazaninHandoverDecision::runProperTGNN() {
    static bool properTGNNStarted = false;
    if (properTGNNStarted) {
        return; // don't spawn again
    }
    properTGNNStarted = true;

    std::string cmd = "python3 " + baseFilePath + "infer_proper_tgnn.py";

    // run in background so OMNeT++ does not block
    cmd += " > /tmp/properTGNN.log 2>&1 &";

    int ret = std::system(cmd.c_str());
    EV_INFO << "[ProperTGNN] Started infer_proper_tgnn.py async, system() returned " << ret << "\n";
}

std::vector<std::pair<int,double>> NazaninHandoverDecision::readProperTGNNOutput(const std::string& filepath)
{
    std::vector<std::pair<int,double>> results;
    std::ifstream in(filepath);
    std::string line;

    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string towerStr, scoreStr;

        if (std::getline(ss, towerStr, ',') && std::getline(ss, scoreStr)) {
            int towerId = std::stoi(towerStr);
            double score = std::stod(scoreStr);
            results.push_back({towerId, score});
        }
    }
    return results;
}

void NazaninHandoverDecision::appendTGNNRow(const TGNNRow& row)
{
    auto& hist = tgnnHistory[row.vehicleId];
    hist.push_back(row);

    while ((int)hist.size() > tgnnSeqLen) {
        hist.pop_front();
    }
}

void NazaninHandoverDecision::writeTGNNRuntimeWindow(int vehicleId, const std::string& filepath)
{
    std::ofstream out(filepath);
    out << "timestamp,vehicleId,masterId,candidateMasterId,masterDistance,candidateDistance,"
           "masterRSSI,candidateRSSI,masterSINR,candidateSINR,masterRSRP,candidateRSRP,"
           "masterSpeed,candidateSpeed,vehicleDirection,vehiclePosition-x,vehiclePosition-y,"
           "towerload\n";

    auto it = tgnnHistory.find(vehicleId);
    if (it == tgnnHistory.end()) return;

    for (const auto& r : it->second) {
        out << r.timestamp << ","
            << r.vehicleId << ","
            << r.masterId << ","
            << r.candidateMasterId << ","
            << r.masterDistance << ","
            << r.candidateDistance << ","
            << r.masterRSSI << ","
            << r.candidateRSSI << ","
            << r.masterSINR << ","
            << r.candidateSINR << ","
            << r.masterRSRP << ","
            << r.candidateRSRP << ","
            << r.masterSpeed << ","
            << r.candidateSpeed << ","
            << r.vehicleDirection << ","
            << r.vehiclePosX << ","
            << r.vehiclePosY << ","
            << r.towerload
            << "\n";
    }
}

void NazaninHandoverDecision::runSVR(unsigned short  vehicleID, int simTime) {

    std::string pypredSVR_CmdPyCpp = "python3 " + baseFilePath + "SVMRegression.py " + std::to_string(vehicleID) +" "+ std::to_string(simTime) ;

    system(pypredSVR_CmdPyCpp.c_str());
}

NazaninHandoverDecision::SpeedCategory NazaninHandoverDecision::getSpeedCategory(double vSpeed) {
    if (vSpeed >= 0 && vSpeed <= 20)
        return SPEED_020;
    else if (vSpeed <= 40)
        return SPEED_2140;
    else if (vSpeed <= 60)
        return SPEED_4160;
    else if (vSpeed <= 80)
        return SPEED_6180;
    else if (vSpeed <= 100)
        return SPEED_81100;
    else if (vSpeed <= 120)
        return SPEED_101120;
    else if (vSpeed <= 140)
        return SPEED_121140;
    else if (vSpeed <= 160)
        return SPEED_141160;
    else
        return SPEED_160PLUS;
}


//void LtePhyUe::performaceAnalysis()
//{
//    std::array<std::string, SPEED_COUNT> speedNames = {
//           "SPEED_020", "SPEED_2140", "SPEED_4160", "SPEED_6180",
//           "SPEED_81100", "SPEED_101120", "SPEED_121140", "SPEED_141160", "SPEED_160PLUS"
//       };
//    if (simTime().dbl() == 100)
//    {
//        std::cout << "Simtime - " << simTime().dbl() << std::endl;
//
//        // Vehicle Count
//        std::cout << "vehicleCount: " << vehicleCountTotal << std::endl;
//
//        // HO count
//        std::cout << "HO count - Num of HO: " << insideVCHO << ", Avg Num of HO: " << insideVCHO / vehicleCountTotal << std::endl;
//
//        // Avg Com HO Time
//        std::cout << "Avg Com HO Time - Com HO Time: " << insideVCHOTime
//                  << ", Avg Com HO Time: " << insideVCHOTime / insideVCHO
//                  << ", Avg HO Time (vehicleCount): " << insideVCHOTime / vehicleCountTotal << std::endl;
//
//        // Each Vehicle HO count
//        std::cout << "Each Vehicle HO count - Vehicle: " << nodeId_ << " - Num of HO: " << vInsideVCHO << std::endl;
//
//        // HO Failure
//        std::cout << "HO Failure - Num of HO Failure: " << fho_insideHOTotal
//                  << ", Avg Num of HO Failure: " << fho_insideHOTotal / insideVCHO << std::endl;
//
//        // HO Ping Pong
//        std::cout << "HO Ping Pong - Num of Ping Pong HO: " << pho_insideHOTotal
//                  << ", Avg Num of Ping Pong HO: " << pho_insideHOTotal / insideVCHO << std::endl;
//
//        // Speed Dependent HO
//        for (int speed = 0; speed < SPEED_COUNT; ++speed) {
//            // Speed Dependent HO
//            std::cout << "Speed Dependent HO - vehicleCount" << speedNames[speed] << ": " << vehicleCount[speed]
//                      << ", insideHO" << speedNames[speed] << ": " << insideHO[speed] << std::endl;
//
//            // Speed Dependent HO Failure
//            std::cout << "Speed Dependent HO Failure -  vehicleCount" << speedNames[speed] << ": " << vehicleCount[speed]
//                      << ", insideHO" << speedNames[speed] << ": " << fho_insideHO[speed] << std::endl;
//
//            // Speed Dependent HO Ping Pong
//            std::cout << "Speed Dependent HO Ping Pong -  vehicleCount" << speedNames[speed] << ": " << vehicleCount[speed]
//                      << ", insideHO" << speedNames[speed] << ": " << pho_insideHO[speed] << std::endl;
//        }
//
//        finishSimTime = simTime().dbl();
//    }
//
//    // saving result at the end of simulation in a file
//    if (simTime().dbl() != finishSimTime && simTime().dbl() > 100)
//    {
//        int result;
//        time_t t = time(0);   // get time now
//        struct tm *now = localtime(&t);
//
//        char buffer[80];
//        strftime(buffer, 80, "%Y%m%d%H%M%S", now);
//        char oldname[] = "/home/nazanin/ProjectHOMng/omnetpp-6.0pre11/batchRunResult/output.txt";
//        std::string str(buffer);
//        std::string strNewFile = "/home/nazanin/ProjectHOMng/omnetpp-6.0pre11/bin/batchRunResult/" + str + ".txt";
//        int len = strNewFile.length();
//        char buffer_new[len + 1];
//        strcpy(buffer_new, strNewFile.c_str());
//        result = rename(oldname, buffer_new);
//    }
//}
//
//
//void LtePhyUe::handleNormalHandover(double rssi, double rsrq, double maxSINR, double maxRSRP, double speedDouble, double distanceDouble, SpeedCategory speedCategory) {
//    candidateMasterId_ = sel_srv_Qvalue_id;
//    oldMasterId_ = candidateMasterId_;
//    candidateMasterRssi_ = rssi;
//    candidateMasterSinr_ = maxSINR;
//    candidateMasterRsrp_ = maxRSRP;
//    candidateMasterDist_ = distanceDouble;
//    candidateMasterSpeed_ = speedDouble;
//
//    hysteresisTh_ = updateHysteresisTh(rssi);
//
//    insideVCHO++;
//    vInsideVCHO++;
//    insideVCHOTime = comHOTime;
//
//    insideHO[speedCategory]++;
//    vinsideHO[speedCategory]++;
//
//    binder_->addHandoverTriggered(nodeId_, masterId_, candidateMasterId_);
//
//    ho_Qvalue = rsrq + (200 - maxSINR);
//    ho_rssi = rssi;
//    ho_sinr = maxSINR;
//    ho_rsrp = maxRSRP;
//    ho_dist = distanceDouble;
//    ho_load = avgLoad;
//    avg_srv_QvalueV.clear();
//
//    if (std::find(last_srv_MasterIdV.begin(), last_srv_MasterIdV.end(), candidateMasterId_) != last_srv_MasterIdV.end()) {
//        pho_insideHOTotal++;
//        pho_insideHO[speedCategory]++;
//    }
//
//    if (!handoverStarter_->isScheduled()) {
//        scheduleAt(simTime() + handoverDelta_, handoverStarter_);
//    }
//}
//
//void LtePhyUe::handleFailureHandover(SpeedCategory speedCategory) {
//    fho_insideHOTotal++;
//    fho_vInsideHO++;
//
//    fho_insideHO[speedCategory]++;
//    fho_vinsideHO[speedCategory]++;
//
//    if (!handoverStarter_->isScheduled()) {
//        scheduleAt(simTime() + handoverDelta_, handoverStarter_);
//    }
//}

//
//void LtePhyUe::updateCurrentMaster(double rssi, double maxSINR, double maxRSRP, double distanceDouble) {
//    currentMasterRssi_ = rssi;
//    currentMasterSinr_ = maxSINR;
//    currentMasterRsrp_ = maxRSRP;
//    currentMasterDist_ = distanceDouble;
//
//    candidateMasterRssi_ = rssi;
//    candidateMasterSinr_ = maxSINR;
//    candidateMasterRsrp_ = maxRSRP;
//    candidateMasterDist_ = distanceDouble;
//}
//

//
//void LtePhyUe::performHysteresisUpdate(double currentMasterRssi_, double currentMasterSinr_, double currentMasterRsrp_, double currentMasterDist_) {
//
//    hysteresisTh_ = updateHysteresisTh(currentMasterRssi_);
//    hysteresisSinrTh_ = updateHysteresisThMinSinr(currentMasterSinr_);
//    hysteresisRsrpTh_ = updateHysteresisThMinRsrp(currentMasterRsrp_);
//    hysteresisDistTh_ = updateHysteresisThMaxDist(currentMasterDist_);
//    hysteresisLoadTh_ = updateHysteresisTowerLoad(avgLoad);
//
//}
//
//
//double NazaninHandoverDecision::updateHysteresisThMinSinr(double v, double hysteresisFactorSinr_)   // v is same as x
//{
//    if (hysteresisFactorSinr_ == 0)
//        return 0;
//    else
//        return ((v / hysteresisFactorSinr_) - 3);
//}
//
//double NazaninHandoverDecision::updateHysteresisThMinRsrp(double v, double hysteresisFactorRsrp_)
//{
//    if (hysteresisFactorRsrp_ == 0)
//        return 0;
//    else
//        return ((v / hysteresisFactorRsrp_) + 20);
//}
//
//double NazaninHandoverDecision::updateHysteresisThMaxDist(double v, double hysteresisFactorDist_)
//{
//    if (hysteresisFactorDist_ == 0)
//        return 0;
//    else
//        return ((v / hysteresisFactorDist_) + 150) ;
//}
//
//double NazaninHandoverDecision::updateHysteresisTowerLoad(double v, double hysteresisFactorLoad_)
//{
//    if (hysteresisFactorLoad_ == 0)
//        return 0;
//    else
//        return ((v / hysteresisFactorLoad_) + 0.5) ;
//}


void NazaninHandoverDecision::calculateReward(double& rewd, double rssi, double avgLoad, double distanceDouble, std::vector<MacNodeId>& last_srv_MasterIdV) {
    rewd = (rssi + avgLoad + (5000 - distanceDouble)) / 3;

}
void NazaninHandoverDecision::calculateTimeInterval(double vIndiSpeed, double& srl_alpha, double& srl_gamma) {
    if ((int)simTime().dbl() % 5 == 0) {
        if (vIndiSpeed > 0 && vIndiSpeed <= 60) {
            srl_alpha = 0.8;
            srl_gamma = 0.8;
        } else if (vIndiSpeed > 61 && vIndiSpeed <= 120) {
            srl_alpha = 0.5;
            srl_gamma = 0.5;
        } else if (vIndiSpeed > 121 && vIndiSpeed <= 180) {
            srl_alpha = 0.3;
            srl_gamma = 0.1;
        } else {
            srl_alpha = 0.3;
            srl_gamma = 0.1;
        }
    }
}

void NazaninHandoverDecision::updateQValue(double& max_Qvalue, double& upt_Qvalue, std::vector<double>& upt_QvalueV, double ho_Qvalue,
                     double srl_alpha, double rewd, double srl_gamma, double srv_Qvalue, double mbr_Qvalue) {
       upt_Qvalue = ho_Qvalue + srl_alpha * (rewd + ((srl_gamma * srv_Qvalue) - mbr_Qvalue));
       upt_QvalueV.push_back(upt_Qvalue);
       max_Qvalue = *max_element(upt_QvalueV.begin(), upt_QvalueV.end());
   }


//void NazaninHandoverDecision::updateCandidate(double scalPara, double predScaValLSTM, MacNodeId& sel_srv_Qvalue_id, MacNodeId& mbr_Qvalue_id, UserControlInfo* lteInfo) {
//    if (scalPara > predScaValLSTM) {
//        sel_srv_Qvalue_id = lteInfo->getSourceId();
//    } else {
//        mbr_Qvalue_id = lteInfo->getSourceId();
//    }
//}
void NazaninHandoverDecision::performHysteresisUpdate(double& hysteresisTh_, double& hysteresisSinrTh_, double& hysteresisRsrpTh_,
        double& hysteresisDistTh_, double& hysteresisLoadTh_, double currentMasterRssi_, double currentMasterSinr_, double currentMasterRsrp_, double currentMasterDist_) {

    hysteresisTh_ = lte->updateHysteresisTh(currentMasterRssi_);
    hysteresisSinrTh_ = lte->updateHysteresisThMinSinr(currentMasterSinr_);
    hysteresisRsrpTh_ = lte->updateHysteresisThMinRsrp(currentMasterRsrp_);
    hysteresisDistTh_ = lte->updateHysteresisThMaxDist(currentMasterDist_);
    hysteresisLoadTh_ = lte->updateHysteresisTowerLoad(avgLoad);

}
