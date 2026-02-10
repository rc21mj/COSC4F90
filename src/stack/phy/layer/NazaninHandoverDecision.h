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

//#include "common/LteControlInfo_m.h"
#include "common/LteControlInfo.h"
//#include "stack/phy/packet/LteAirFrame_m.h"
#include "stack/phy/ChannelModel/LteChannelModel.h"


#include <assert.h>

#include <inet/common/INETDefs.h>
#include <ctime>
#include <vector>
#include <numeric>

#ifndef STACK_PHY_LAYER_NAZANINHANDOVERDECISION_H_
#define STACK_PHY_LAYER_NAZANINHANDOVERDECISION_H_



class NazaninHandoverDecision {
public:
    NazaninHandoverDecision();
    virtual ~NazaninHandoverDecision();

    inet::Coord VehiclePosCal;
    inet::Coord towerPosition;
    //
    //
    static const int NUM_TOWERS = 10;
    double bsLoad[NUM_TOWERS] = {0};
        enum SpeedCategory {
            SPEED_020, SPEED_2140, SPEED_4160, SPEED_6180, SPEED_81100, SPEED_101120, SPEED_121140, SPEED_141160, SPEED_160PLUS, SPEED_COUNT
        };
        std::map<int, std::pair<double, double> >  Tower_Position = { { 1, {834,1041}}, { 2, {2496,1155} }, { 3, {4293,1155} },{ 4, {734,2389} },{ 5, {2225,2289} },{ 6, {3566,2389} },{ 7, {1233,3580} },{ 8, {2391,3372} },{ 9, {3162,3672} },{ 10, {4393,2995} } };
    //
    //
        std::tuple<double, double, double, double> calculateMetrics(LteChannelModel* primaryChannelModel_, LteAirFrame* frame, UserControlInfo* lteInfo);
        double getParfromFile(std::string filepath);
        std::tuple<double, double> getParfromFileForSVR(std::string filepath);
        void calculateTowerLoad(UserControlInfo* lteInfo, LteAirFrame* frame);
        double towerLoad(LteAirFrame* frame, UserControlInfo* lteInfo);
        void saveParaToFile(std::string filepath, double para);
        void saveStringParaToFile(std::string filepath,std::string   para);
        void saveArrayToFile(const std::string& fileName, const std::vector<double>& array);
        void runLSTM();
        void runTGNN();
        void runTGNNdiff();
        void runSVR(unsigned short vehicleID, int simTime);
        SpeedCategory getSpeedCategory(double vSpeed);
        void calculateReward(double& rewd, double rssi, double avgLoad, double distanceDouble, std::vector<MacNodeId>& last_srv_MasterIdV);
        void calculateTimeInterval(double vIndiSpeed, double& srl_alpha, double& srl_gamma);
        void updateQValue(double& max_Qvalue, double& upt_Qvalue, std::vector<double>& upt_QvalueV, double ho_Qvalue,
                             double srl_alpha, double rewd, double srl_gamma, double srv_Qvalue, double mbr_Qvalue);
        void updateCandidate(double scalPara, double predScaValLSTM, MacNodeId& sel_srv_Qvalue_id, MacNodeId& mbr_Qvalue_id, UserControlInfo* lteInfo);
//        void performaceAnalysis();
    //    void handleNormalHandover(double rssi, double rsrq, double maxSINR, double maxRSRP, double speedDouble, double distanceDouble, SpeedCategory speedCategory);
    //    void updateCurrentMaster(double rssi, double maxSINR, double maxRSRP, double distanceDouble);
    //    void handleFailureHandover(SpeedCategory speedCategory);
        void performHysteresisUpdate(double& hysteresisTh_, double& hysteresisSinrTh_, double& hysteresisRsrpTh_,
                double& hysteresisDistTh_, double& hysteresisLoadTh_, double currentMasterRssi_, double currentMasterSinr_, double currentMasterRsrp_, double currentMasterDist_);

        std::vector<int> GetClosestTowersId(double xCoordRef, double yCoordRef, int vehicleID, int curTower);
//        std::vector<double>speedV;
//        double sumSpeedV, avgSpeed;
//
//        MacNodeId dirMacNodeId;
//
//        //q value
//        std::vector<double> upt_QvalueV;
//        std::vector<double> avg_srv_QvalueV;
//        double upt_Qvalue = 1, srv_Qvalue = 1, mbr_Qvalue = 1, ho_Qvalue = 1, rewd = 1, max_Qvalue = 1, avg_srv_Qvalue = 1;
//        MacNodeId upt_Qvalue_id = 1, srv_Qvalue_id = 1, mbr_Qvalue_id = 1, ho_Qvalue_id = 1, max_Qvalue_id = 1, sel_srv_Qvalue_id = 1;
//        double upt_srv_Qvalue = 1, upt_mbr_Qvalue = 1;
//        double temp_upt_Qvalue = 1;
//        double srl_alpha = 1, srl_gamma = 1;
//
//        double ho_rssi = 10, ho_sinr = 10, ho_rsrp = -50, ho_dist = 1000, ho_load = 1;
//
//        double vInsideVCHO = 0, vOutsideVCHO = 0, vCurTowerVCHO = 0, fho_vInsideHO = 0;
//        double vvehicleCount020 = 0, vvehicleCount2140 = 0, vvehicleCount4160 = 0, vvehicleCount6180 = 0, vvehicleCount81100 = 0,
//                vvehicleCount101120 = 0, vvehicleCount121140 = 0, vvehicleCount141160 = 0, vvehicleCount161 = 0, vvehicleCount160plus = 0;
//
//        double vinsideHO020 = 0, fho_vinsideHO020 = 0, vinsideHO2140 = 0, fho_vinsideHO2140 = 0, vinsideHO4160 = 0, fho_vinsideHO4160 = 0,
//                vinsideHO6180 = 0, fho_vinsideHO6180 = 0, vinsideHO81100 = 0, fho_vinsideHO81100 = 0, vinsideHO101120 = 0, fho_vinsideHO101120 = 0,
//                vinsideHO121140 = 0, fho_vinsideHO121140 = 0, vinsideHO141160 = 0, fho_vinsideHO141160 = 0, vinsideHO160plus = 0, fho_vinsideHO160plus = 0;
//
//        double vIndiSpeed = 0;
//
//        double scalPara = 0, predScaValLSTM = 0;
//
//        MacNodeId oldMasterId_ = 1;
//        std::vector<MacNodeId> last_srv_MasterIdV;
//
//        std::string baseFilePath = "/home/nazanin/ProjectHOMng/simu5G/src/stack/phy/layer/";
//        std::string speedFile = "/home/nazanin/ProjectHOMng/simu5G/src/stack/phy/ChannelModel/speedFile.txt";

};

#endif /* STACK_PHY_LAYER_NAZANINHANDOVERDECISION_H_ */
