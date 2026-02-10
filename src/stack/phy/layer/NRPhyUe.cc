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

#include "stack/phy/layer/NRPhyUe.h"

#include "stack/ip2nic/IP2Nic.h"
#include "stack/phy/feedback/LteDlFeedbackGenerator.h"
#include "stack/d2dModeSelection/D2DModeSelectionBase.h"
#include "stack/phy/layer/LtePhyUe.h"
#include "stack/phy/layer/NazaninHandoverDecision.h"
#include <numeric>

using namespace std;

NazaninHandoverDecision* HoMngNR = new NazaninHandoverDecision();

//previous work FiVH (1)

//simtime
double bsdeg_cur_simtime = 1;   //for bsdeg
double pre_betw_cur_simtime = 1; //for pre betw using nst/gst
double betw_cur_simtime = 1;  //for betw
double minDist_cur_simtime = 1; //for min dist

//for tower selection at the end of pw VC
MacNodeId N1, T1, N2, T2, N3, T3;

std::map<MacNodeId, MacNodeId> map_bsdeg_selectedTowerVehilce;
std::map<MacNodeId, MacNodeId> map_betw_selectedTowerVehilce;
std::map<MacNodeId, MacNodeId> map_minDist_selectedTowerVehilce;
std::map<MacNodeId, MacNodeId>::iterator find_it1;
std::map<MacNodeId, MacNodeId>::iterator find_it2;
std::map<MacNodeId, MacNodeId>::iterator find_it3;

//bsdeg
double BSsize1;
double BSsize2;
double BSsize3;
double BSsize4;
double BSsize5;
double BSdeg1;
double BSdeg2;
double BSdeg3;
double BSdeg4;
double BSdeg5;
MacNodeId bsdeg_towerId;

std::map<MacNodeId, std::map<MacNodeId, double>> map_bsdeg;   //min bsdeg for each node and for each tower
std::map<MacNodeId, std::map<MacNodeId, double>>::iterator map_bsdeg_itr;
std::map<MacNodeId, double>::iterator map_bsdeg_ptr;

//min dist
double minDist1;
double minDist2;
double minDist3;
double minDist4;
double minDist5;
MacNodeId minDist_towerId;

std::map<MacNodeId, std::map<MacNodeId, double>> map_minDist;   //min dist for each node and for each tower
std::map<MacNodeId, std::map<MacNodeId, double>>::iterator map_minDist_itr;
std::map<MacNodeId, double>::iterator map_minDist_ptr;

//betw
double gstSqrDistance;
double pBetw1;
double pBetw2;
double pBetw3;
double pBetw4;
double pBetw5;
MacNodeId betw_towerId;

std::map<MacNodeId, std::map<MacNodeId, double>> map_betw;   //min dist for each node and for each tower
std::map<MacNodeId, std::map<MacNodeId, double>>::iterator map_betw_itr;
std::map<MacNodeId, double>::iterator map_betw_ptr;

//end of previous work FiVH (1)
MacNodeId dirTower = 1;
long double x[5], y[5], sumX = 0, sumX2 = 0, sumY = 0, sumXY = 0, a = 0, b = 0;
double time1, time2, time3, time4, time5;
double lastTimeCount;
double HOTimeNR, comHOTimeNR;
int counter_handleairframe=0;
long double difCoordVTy, tempdifCoordVTy;

std::string csvFilePathForVehicle = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/VehicleMovementTracker.csv";

Define_Module(NRPhyUe);

NRPhyUe::NRPhyUe()
{
    handoverStarter_ = NULL;
    handoverTrigger_ = NULL;
}

NRPhyUe::~NRPhyUe()
{
}

void NRPhyUe::initialize(int stage)
{
    LtePhyUeD2D::initialize(stage);
    if (stage == inet::INITSTAGE_LOCAL)
    {
        isNr_ = (strcmp(getFullName(),"nrPhy") == 0) ? true : false;
        if (isNr_)
        {
            otherPhy_ = check_and_cast<NRPhyUe*>(getParentModule()->getSubmodule("phy"));
        }
        else
        {
            otherPhy_ = check_and_cast<NRPhyUe*>(getParentModule()->getSubmodule("nrPhy"));
        }
    }
}

void NRPhyUe::calculatePredDirection(NRPhyUe* otherPhy_, UserControlInfo* lteInfo) {

    n = 5;

      if (simTime().dbl() == (time1 * 5) + 1)
      {
          //initial them by assigning 0 to them
          sumX = sumY = sumX2 = sumXY = 0;

          x[1] = otherPhy_->getCoord().x;
          y[1] = otherPhy_->getCoord().y;

          //assigning value for each simtime in array[i] position, 5 simtime will put 5 values
          sumX = sumX + x[1];
          sumX2 = sumX2 + (x[1] * x[1]);
          sumY = sumY + y[1];
          sumXY = sumXY + (x[1] * y[1]);
  //        std::cout << "1 sumX " << sumX << " sumY " << sumY << " sumX2 " << sumX2 << " sumXY " << sumXY << endl;

          //incrementing time1++ for triggering in a specific simtime
          time1++;

          //to check whether the value has come in the mid of simtime cycle, if simtimeAdjstNum == 0; came in the mid of the cycle
          simtimeAdjstNum++;
          lastTimeCount = time1;
      }

      //assigning time cycle count number to a vehicle that comes in the mid of a cycle
      if (simtimeAdjstNum == 0)
      {
  //        std::cout << "node id " << nodeId_ << " lastTimeCount " << lastTimeCount << endl;
          time1 = time2 = time3 = time4 = time5 = lastTimeCount;
      }

      if (simTime().dbl() == (time2 * 5) + 2 && simtimeAdjstNum != 0)
      {
          x[2] = otherPhy_->getCoord().x;
          y[2] = otherPhy_->getCoord().y;

          sumX = sumX + x[2];
          sumX2 = sumX2 + (x[2] * x[2]);
          sumY = sumY + y[2];
          sumXY = sumXY + (x[2] * y[2]);
  //        std::cout << "2 sumX " << sumX << " sumY " << sumY << " sumX2 " << sumX2 << " sumXY " << sumXY << endl;

          time2++;
      }
      if (simTime().dbl() == (time3 * 5) + 3 && simtimeAdjstNum != 0)
      {
          x[3] = otherPhy_->getCoord().x;
          y[3] = otherPhy_->getCoord().y;

          sumX = sumX + x[3];
          sumX2 = sumX2 + (x[3] * x[3]);
          sumY = sumY + y[3];
          sumXY = sumXY + (x[3] * y[3]);
  //        std::cout << "3 sumX " << sumX << " sumY " << sumY << " sumX2 " << sumX2 << " sumXY " << sumXY << endl;

          time3++;
      }
      if (simTime().dbl() == (time4 * 5) + 4 && simtimeAdjstNum != 0)
      {
          x[4] = otherPhy_->getCoord().x;
          y[4] = otherPhy_->getCoord().y;

          sumX = sumX + x[4];
          sumX2 = sumX2 + (x[4] * x[4]);
          sumY = sumY + y[4];
          sumXY = sumXY + (x[4] *  y[4]);
  //        std::cout << "4 sumX " << sumX << " sumY " << sumY << " sumX2 " << sumX2 << " sumXY " << sumXY << endl;

          time4++;
      }
      if (simTime().dbl() == (time5 * 5) + 5 && simtimeAdjstNum != 0)
      {
          x[5] = otherPhy_->getCoord().x;
          y[5] = otherPhy_->getCoord().y;

          sumX = sumX + x[5];
          sumX2 = sumX2 + (x[5] * x[5]);
          sumY = sumY + y[5];
          sumXY = sumXY + (x[5] * y[5]);
  //        std::cout << "5 sumX " << sumX << " sumY " << sumY << " sumX2 " << sumX2 << " sumXY " << sumXY << endl;

          time5++;

          //cal. of y = mx + c for the last 5 simtime
          //Calculating a(m), b(c) for end of each simtime cycle (5 sec)
          b = (n * sumXY - sumX * sumY ) / (n  * sumX2 - sumX * sumX);    //m
          a = (sumY - b * sumX) / n;  //c

  //        std::cout << "Vehicle: " << nodeId_ << " inside b: " << b << " a: " << a << endl;
      }

      //Displaying value of a and b in each simtime
  //    std::cout << "Vehicle: " << nodeId_ << " outside b: " << b << " a: " << a << endl;

      double coordVx = otherPhy_->getCoord().x;
      double coordVy = otherPhy_->getCoord().y;
      double vehicle_ID = otherPhy_->nodeId_;
      std::string vehicleInfo = to_string(simTime().dbl()) +  "   " + to_string(nodeId_) + "    "+   to_string(coordVx) + "    "+  to_string(coordVy);





     // HoMngNR->saveStringParaToFile("VehicleMovementTrack.txt", vehicleInfo);
      //std::cout << "Shajib::NRPhyUe::calculatePredDirection VEHICLE INFO:" <<  vehicleInfo << std::endl;

     // std::cout << "Shajib::NRPhyUe::calculatePredDirection VEHICLE ID: " << nodeId_ << " - TOWER ID: " <<  lteInfo->getSourceId() << " X Coord: " << coordVx << " Y Coord: "<< coordVy << std::endl;

      //std::cout << "Shajib::NRPhyUe::calculatePredDirection Sim Time: " << simTime().dbl()) << "VEHICLE ID: " << nodeId_ << " - TOWER ID: " <<  lteInfo->getSourceId() << " X Coord: " << coordVx << " Y Coord: "<< coordy <<  std::endl;



          double speedOfVehicle = computeSpeed(nodeId_, otherPhy_->getCoord());
         // double speedOfVehicle = HoMngNR->getParfromFile("/home/shajib/Simulation/Myversion2/WorkFolder/simu5G/src/stack/phy/ChannelModel/speedFile.txt");
          CSVRowForVehiclePosition newRow;

            // Set values for the members of the CSVRow struct
            newRow.vehicleId = nodeId_;
            newRow.speed = speedOfVehicle;
            newRow.towerId = lteInfo->getSourceId();
            newRow.timestamp = simTime().dbl();
            newRow.vehiclePositionx = coordVx;
            newRow.vehiclePositiony = coordVy;

            // Add the new row to the CSV file

            addRowToCSV(csvFilePathForVehicle, newRow);
            //std::cout << "Shajib::NRPhyUe::calculatePredDirection Row added to CSV file.\n";



      double preCoordVy = (b * coordVx) + a;
  //    std::cout << "original coordVx " << coordVx << endl;
  //    std::cout << "original coordVy " << coordVy << endl;
  //    std::cout << "predicted preCoordVy " << preCoordVy << endl;

      double coordTx = lteInfo->getCoord().x;
      double coordTy = lteInfo->getCoord().y;
      double preCoordTy = (b * coordTx) + a;
  //    std::cout << "Tower: " << lteInfo->getSourceId() << " original coordTx " << coordTx << endl;
  //    std::cout << "Tower: " << lteInfo->getSourceId() << " original coordTy " << coordTy << endl;
  //    std::cout << "Tower: " << lteInfo->getSourceId() << " predicted preCoordTy " << preCoordTy << endl;

      difCoordVTy = abs(preCoordVy - preCoordTy);
//      std::cout << "difCoordVTy: "<< difCoordVTy << " tempdifCoordVTy: " << tempdifCoordVTy << std::endl;
      //cal the predicted dir for a vechile
      if (difCoordVTy < tempdifCoordVTy){
//          std::cout << dirTower << std::endl;
          dirTower = lteInfo->getSourceId();
//          std::cout << dirTower << std::endl;
      }
     // std::cout << "Vehicle: " << nodeId_ << " Dir to Tower: " << dirTower << endl;
      //die calculation using file

      //storing the immediate previous dif of vehicle-tower
      tempdifCoordVTy = difCoordVTy;

      updatePositionHistory(nodeId_, otherPhy_->getCoord());
}

//Shajib:: Custom
void NRPhyUe::addRowToCSV(std::string& filename, const CSVRowForVehiclePosition & newRow) {
  std::fstream file(filename, std::ios::out | std::ios::app);

  if (!file.is_open()) {
      std::cerr << "Error: Unable to open the file for writing." << std::endl;
      return;
  }


  // Write the new row to the file
  file << newRow.timestamp << ',' << newRow.speed << ',' << newRow.vehicleId << ',' << newRow.towerId << ',' << newRow.vehiclePositionx<<','<<newRow.vehiclePositiony<< '\n';

  file.close();
}
//Shajib:: Custom
double NRPhyUe::computeSpeed(const MacNodeId nodeId, const Coord coord)
{

   double speed = 0.0;

   if (positionHistory_.find(nodeId) == positionHistory_.end())
   {
       // no entries
       return speed;
   }
   else
   {
       //compute distance traveled from last update by UE (eNodeB position is fixed)

       if (positionHistory_[nodeId].size() == 1)
       {
           //  the only element refers to present , return 0
           return speed;
       }

       double movement = positionHistory_[nodeId].front().second.distance(coord);

       if (movement <= 0.0)
       {
           return speed;
       }
       else
       {
           double time = (NOW.dbl()) - (positionHistory_[nodeId].front().first.dbl());
           if (time <= 0.0) // time not updated since last speed call
               throw cRuntimeError("Multiple entries detected in position history referring to same time");

           // compute speed
           speed = (movement) / (time);
//           double globalSpeed = speed;
//           std::cout << "vehicle id: " << nodeId << ", coordinates: " << coord << " ,speed: " << speed << std::endl;

       }
   }

   return speed;
}

//Shajib:: Custom
void NRPhyUe::updatePositionHistory(const MacNodeId nodeId, const inet::Coord coord)
{
   if (positionHistory_.find(nodeId) != positionHistory_.end())
   {
       // position already updated for this TTI.
       if (positionHistory_[nodeId].back().first == NOW)
           return;
   }

   // FIXME: possible memory leak
   positionHistory_[nodeId].push(Position(NOW, coord));

   if (positionHistory_[nodeId].size() > 2) // if we have more than a past and a current element
       // drop the oldest one
       positionHistory_[nodeId].pop();
}

// TODO: ***reorganize*** method
void NRPhyUe::handleAirFrame(cMessage* msg)
{
//   std::cout << "<<<<< NRPhyUe::Test Nazanin Packet >>>>>" << msg->getName() << endl;

    UserControlInfo* lteInfo = check_and_cast<UserControlInfo*>(msg->removeControlInfo());
//    std::cout << "BEGINING OF NRPhyUe::handleAirFrame -> lteInfo->getSourceId()" << lteInfo->getSourceId()<< "dir tower" << dirTower<< std::endl;
    if (useBattery_)
    {
        //TODO BatteryAccess::drawCurrent(rxAmount_, 0);
    }
    connectedNodeId_ = masterId_;
    LteAirFrame* frame = check_and_cast<LteAirFrame*>(msg);

    int sourceId = lteInfo->getSourceId();
    if(binder_->getOmnetId(sourceId) == 0 )
    {
        // source has left the simulation
        delete msg;
//        std::cout << "END OF NRPhyUe::handleAirFrame -> lteInfo->getSourceId()" << lteInfo->getSourceId()<< "dir tower" << dirTower<< std::endl;
        return;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // declare and call value/data from other file by declaring as object         //
    // and object name same as class name                                         //
    // object->function to call                                                   //
    // e.g. LteRealisticChannelModel* LteRealChannelModel;                        //
    ////////////////////////////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////  Virtual Cell (VC)  ////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////


    if (nodeId_ >= 2000) //taking only vehicles
    {

    auto result = HoMngNR->calculateMetrics(primaryChannelModel_, frame, lteInfo);

     double rssi = std::get<0>(result);
     double maxSINR = std::get<1>(result);
     double maxRSRP = std::get<2>(result);

//     std::cout << "RSSI: " << rssi << " Vehicle: "<< nodeId_ << " Tower " << lteInfo->getSourceId() << endl;
//
//     std::cout << "SINR: " << maxSINR << " Vehicle: "<< nodeId_ << " Tower " << lteInfo->getSourceId() << endl;
//
//     std::cout << "RSRP: " << maxRSRP << " Vehicle: "<< nodeId_ << " Tower " << lteInfo->getSourceId() << endl;


    //distance between vehicle and tower
    double sqrDistance = otherPhy_->getCoord().distance(lteInfo->getCoord());


   // std::cout << "VEHICLE ID: " << nodeId_ << " POSITION: "<< vehiclePosition << std::endl;


    //Print distance between vehicle and tower
//   std::cout << "NRPhyUe::handleAirFrame Distance: " << sqrDistance << " Vehicle: "<< nodeId_ << " Tower " << lteInfo->getSourceId() << endl;
//   std::cout << "lteInfo->getCoord(): " << lteInfo->getCoord() << std::endl;
    //distance calculation using file
    double globalDistance = sqrDistance;
    HoMngNR->saveParaToFile("distanceFile.txt", globalDistance);

    // saving the position of vehicle in a file
    Coord vehiclePosition = otherPhy_->getCoord();
    std::ofstream file(baseFilePath+"vehiclePositionFile.txt", std::ofstream::trunc);
    file << vehiclePosition.x << ' ' << vehiclePosition.y << ' ' << vehiclePosition.z;
    file.close();


    //end comment when run pw (22)
    double speedDouble = HoMngNR->getParfromFile("/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/ChannelModel/speedFile.txt");

//    std::cout << "FIRST -> NRPhyUe::handleAirFrame dirTower: " << dirTower << endl;

    calculatePredDirection(otherPhy_, lteInfo);


//    std::cout << "SECOND -> NRPhyUe::handleAirFrame dirTower: " << dirTower << endl;

    // Save parameters to file
    HoMngNR->saveParaToFile("dirFile.txt", dirTower);


//    std::cout << "frame->getLossRate() " << frame->getLossRate() << endl;


    } //end of taking only vehicles

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    double carrierFreq = lteInfo->getCarrierFrequency();

    LteChannelModel* channelModel = getChannelModel(carrierFreq);
    if (channelModel == NULL)
    {

        //EV << "Received packet on carrier frequency not supported by this node. Delete it." << endl;
        delete lteInfo;
        delete frame;
        return;
    }
  //  std::cout<< "Shajib:: Checking NRPhyUe::handleAirFrame"<< endl;
    //Update coordinates of this user
    if (lteInfo->getFrameType() == HANDOVERPKT)
    {
        //std::cout<< "Shajib:: Checking Handover"<< endl;
        // check if the message is on another carrier frequency or handover is already in process
        if (carrierFreq != primaryChannelModel_->getCarrierFrequency() || (handoverTrigger_ != NULL && handoverTrigger_->isScheduled()))
        {
            //EV << "Received handover packet on a different carrier frequency. Delete it." << endl;
            delete lteInfo;
            delete frame;
            return;
        }

        // check if the message is from a different cellular technology
        if (lteInfo->isNr() != isNr_)
        {
            //EV << "Received handover packet [from NR=" << lteInfo->isNr() << "] from a different radio technology [to NR=" << isNr_ << "]. Delete it." << endl;
            delete lteInfo;
            delete frame;
            return;
        }

        // check if the eNodeB is a secondary node
        MacNodeId masterNodeId = binder_->getMasterNode(sourceId);
        if (masterNodeId != sourceId)
        {
            // the node has a master node, check if the other PHY of this UE is attached to that master.
            // if not, the UE cannot attach to this secondary node and the packet must be deleted.
            if (otherPhy_->getMasterId() != masterNodeId)
            {
                //EV << "Received handover packet from " << sourceId << ", which is a secondary node to a master [" << masterNodeId << "] different from the one this UE is attached to. Delete packet." << endl;
                delete lteInfo;
                delete frame;
                return;
            }
        }

        //comment when run pw (11)


        handoverHandler(frame, lteInfo);
        //end comment when run pw (11)

        return;
    }

    // Check if the frame is for us ( MacNodeId matches or - if this is a multicast communication - enrolled in multicast group)
    if (lteInfo->getDestId() != nodeId_ && !(binder_->isInMulticastGroup(nodeId_, lteInfo->getMulticastGroupId())))
    {
        //EV << "ERROR: Frame is not for us. Delete it." << endl;
        //EV << "Packet Type: " << phyFrameTypeToA((LtePhyFrameType)lteInfo->getFrameType()) << endl;
        //EV << "Frame MacNodeId: " << lteInfo->getDestId() << endl;
        //EV << "Local MacNodeId: " << nodeId_ << endl;
        delete lteInfo;
        delete frame;
        return;
    }

    /*
     * This could happen if the ue associates with a new master while a packet from the
     * old master is in-flight: the packet is in the air
     * while the ue changes master.
     * Event timing:      TTI x: packet scheduled and sent by the UE (tx time = 1ms)
     *                     TTI x+0.1: ue changes master
     *                     TTI x+1: packet from UE arrives at the old master
     */
    if (lteInfo->getDirection() != D2D && lteInfo->getDirection() != D2D_MULTI && lteInfo->getSourceId() != masterId_)
    {
        //EV << "WARNING: frame from a UE that is leaving this cell (handover): deleted " << endl;
        //EV << "Source MacNodeId: " << lteInfo->getSourceId() << endl;
        //EV << "UE MacNodeId: " << nodeId_ << endl;
        delete lteInfo;
        delete frame;
        return;
    }

    if (binder_->isInMulticastGroup(nodeId_,lteInfo->getMulticastGroupId()))
    {
        // HACK: if this is a multicast connection, change the destId of the airframe so that upper layers can handle it
        lteInfo->setDestId(nodeId_);
    }

    // send H-ARQ feedback up
    if (lteInfo->getFrameType() == HARQPKT || lteInfo->getFrameType() == GRANTPKT || lteInfo->getFrameType() == RACPKT || lteInfo->getFrameType() == D2DMODESWITCHPKT)
    {
        handleControlMsg(frame, lteInfo);
        return;
    }

    // this is a DATA packet

    // if the packet is a D2D multicast one, store it and decode it at the end of the TTI
    if (d2dMulticastEnableCaptureEffect_ && binder_->isInMulticastGroup(nodeId_,lteInfo->getMulticastGroupId()))
    {
        // if not already started, auto-send a message to signal the presence of data to be decoded
        if (d2dDecodingTimer_ == NULL)
        {
            d2dDecodingTimer_ = new cMessage("d2dDecodingTimer");
            d2dDecodingTimer_->setSchedulingPriority(10);          // last thing to be performed in this TTI
            scheduleAt(NOW, d2dDecodingTimer_);
        }

        // store frame, together with related control info
        frame->setControlInfo(lteInfo);
        storeAirFrame(frame);            // implements the capture effect

        return;                          // exit the function, decoding will be done later
    }

    if ((lteInfo->getUserTxParams()) != NULL)
    {
        int cw = lteInfo->getCw();
        if (lteInfo->getUserTxParams()->readCqiVector().size() == 1)
            cw = 0;
        double cqi = lteInfo->getUserTxParams()->readCqiVector()[cw];
        if (lteInfo->getDirection() == DL)
        {
            emit(averageCqiDl_, cqi);
            recordCqi(cqi, DL);
        }
    }
    // apply decider to received packet
    bool result = true;
    RemoteSet r = lteInfo->getUserTxParams()->readAntennaSet();
    if (r.size() > 1)
    {
        // DAS
        for (RemoteSet::iterator it = r.begin(); it != r.end(); it++)
        {
            //EV << "NRPhyUe: Receiving Packet from antenna " << (*it) << "\n";

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
        result = channelModel->isErrorDas(frame,lteInfo);
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

    //EV << "Handled LteAirframe with ID " << frame->getId() << " with result "
//       << ( result ? "RECEIVED" : "NOT RECEIVED" ) << endl;

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

// never called
double NRPhyUe::NrDistCal(cMessage* msg, Coord coord)
{
    UserControlInfo* lteInfoDist = check_and_cast<UserControlInfo*>(msg->removeControlInfo());

    //distance between vehicle and tower
    double NrSqrDistance = lteInfoDist->getCoord().distance(coord);

    return NrSqrDistance;
}

// never called
double NRPhyUe::NrSinrCal(cMessage* msg)
{
    UserControlInfo* lteInfoSinr = check_and_cast<UserControlInfo*>(msg->removeControlInfo());
    LteAirFrame* frameSinr = check_and_cast<LteAirFrame*>(msg);

    //SINR and RSSI
    std::vector<double>::iterator it;
    double rssi = 0;

    //taking value for RSSI
    std::vector<double> rssiV = primaryChannelModel_->getSINR(frameSinr, lteInfoSinr);  //getting SINR, rssiV - SINR vector

    for (it = rssiV.begin(); it != rssiV.end(); ++it)
        rssi += *it;
    rssi /= rssiV.size();   //getting RSSI

    //taking value for Max SINR
    double maxSINR = *max_element(rssiV.begin(), rssiV.end()); //getting MAX SINR
//    std::cout << "NRPhyUe:: Max SINR for " << lteInfoSinr->getSourceId() << " is " << maxSINR << endl;

    //Print RSSI
//    std::cout << "NRPhyUe:: RSSI for " << lteInfoSinr->getSourceId() << " is " << rssi << endl;

    return maxSINR;
}

void NRPhyUe::triggerHandover()
{
    // TODO: remove asserts after testing
    assert(masterId_ != candidateMasterId_);

//    std::cout << "##### Handover Starting #####" << endl;
//    std::cout << "Vehicle ID: " << nodeId_ << endl;
//    std::cout << "Current Master ID: " << masterId_ << endl;
//    std::cout << "Candidate Master ID: " << candidateMasterId_ << endl;
//    std::cout << "##########" << endl;

    MacNodeId masterNode = binder_->getMasterNode(candidateMasterId_);
    if (masterNode != candidateMasterId_)  // the candidate is a secondary node, if it was equal it meant candidateMasterID_ IS A MASTER ITSELF
    {
        if (otherPhy_->getMasterId() == masterNode)
        {
            MacNodeId otherNodeId = otherPhy_->getMacNodeId();
            const std::pair<MacNodeId, MacNodeId>* handoverPair = binder_->getHandoverTriggered(otherNodeId);
            if (handoverPair != NULL)
            {
                if (handoverPair->second == candidateMasterId_)
                {
                    // delay this handover
                    double delta = handoverDelta_;
                    if (handoverPair->first != 0) // the other "stack" is performing a complete handover
                        delta += handoverDetachment_ + handoverAttachment_;
                    else                          // the other "stack" is attaching to an eNodeB
                        delta += handoverAttachment_;

                    //EV << NOW << " NRPhyUe::triggerHandover - Wait the handover completion for the other stack. Delay this handover." << endl;

                    // need to wait for the other stack to complete handover
                    scheduleAt(simTime() + delta, handoverStarter_);
                    return;
                }
                else
                {
                    // cancel this handover
                    binder_->removeHandoverTriggered(nodeId_);
                    //EV << NOW << " NRPhyUe::triggerHandover - UE " << nodeId_ << " is canceling its handover to gNB " << candidateMasterId_ << " since the master is performing handover" << endl;
                    return;
                }
            }
        }
    }
    // else it is a master itself


    if (candidateMasterRssi_ == 0)
    {
        //EV << NOW << " NRPhyUe::triggerHandover - UE " << nodeId_ << " lost its connection to gNB " << masterId_ << ". Now detaching... " << endl;
//        std::cout << NOW << " NRPhyUe::triggerHandover - UE " << nodeId_ << " lost its connection to gNB " << masterId_ << ". Now detaching... " << endl;
    }
    else if (masterId_ == 0)
    {
        //EV << NOW << " NRPhyUe::triggerHandover - UE " << nodeId_ << " is starting attachment procedure to gNB " << candidateMasterId_ << "... " << endl;
//        std::cout << " NRPhyUe::triggerHandover - UE " << nodeId_ << " is starting attachment procedure to gNB " << candidateMasterId_ << "... " << endl;
    }
    else
    {
        //EV << NOW << " NRPhyUe::triggerHandover - UE " << nodeId_ << " is starting handover to gNB " << candidateMasterId_ << "... " << endl;
//        std::cout << NOW << " NRPhyUe::triggerHandover - UE " << nodeId_ << " is starting handover to gNB " << candidateMasterId_ << "... " << endl;
    }
    if (otherPhy_->getMasterId() != 0)
    {
        // check if there are secondary nodes connected
        MacNodeId otherMasterId = binder_->getMasterNode(otherPhy_->getMasterId());
        if (otherMasterId == masterId_)
        {
            //EV << NOW << " NRPhyUe::triggerHandover - Forcing detachment from " << otherPhy_->getMasterId() << " which was a secondary node to " << masterId_ << ". Delay this handover." << endl;

            // need to wait for the other stack to complete detachment
            scheduleAt(simTime() + handoverDetachment_ + handoverDelta_, handoverStarter_);

            // the other stack is connected to a node which is a secondary node of the master from which this stack is leaving
            // trigger detachment (handover to node 0)
            otherPhy_->forceHandover();

            return;
        }
    }

    binder_->addUeHandoverTriggered(nodeId_);

    // inform the UE's IP2Nic module to start holding downstream packets
    IP2Nic* ip2nic =  check_and_cast<IP2Nic*>(getParentModule()->getSubmodule("ip2nic"));
    ip2nic->triggerHandoverUe(candidateMasterId_, isNr_);

    // inform the eNB's IP2Nic module to forward data to the target eNB
    if (masterId_ != 0 && candidateMasterId_ != 0)
    {
        IP2Nic* enbIp2nic =  check_and_cast<IP2Nic*>(getSimulation()->getModule(binder_->getOmnetId(masterId_))->getSubmodule("cellularNic")->getSubmodule("ip2nic"));
        enbIp2nic->triggerHandoverSource(nodeId_,candidateMasterId_);
    }

    if (masterId_ != 0)
    {
        // stop active D2D flows (go back to Infrastructure mode)
        // currently, DM is possible only for UEs served by the same cell

        // trigger D2D mode switch
        cModule* enb = getSimulation()->getModule(binder_->getOmnetId(masterId_));
        D2DModeSelectionBase *d2dModeSelection = check_and_cast<D2DModeSelectionBase*>(enb->getSubmodule("cellularNic")->getSubmodule("d2dModeSelection"));
        d2dModeSelection->doModeSwitchAtHandover(nodeId_, false);
    }

    double handoverLatency;
    if (masterId_ == 0)                        // attachment only
        handoverLatency = handoverAttachment_;
    else if (candidateMasterId_ == 0)          // detachment only
        handoverLatency = handoverDetachment_;
    else                                       // complete handover time
        handoverLatency = handoverDetachment_ + handoverAttachment_;

    HOTimeNR = handoverLatency;
//    std::cout << "HO Time: " << HOTimeNR << endl;
    comHOTimeNR = comHOTimeNR + HOTimeNR;
//    std::cout << "Commutative HO Time: " << comHOTimeNR << endl;

    //average HO time count
    HoMngNR->saveParaToFile("timeHO.txt", HOTimeNR);
    //commutative HO time count
    double comHOTimeCount = comHOTimeNR;
    HoMngNR->saveParaToFile("comHOTimeCount.txt", comHOTimeCount);

    handoverTrigger_ = new cMessage("handoverTrigger");

    scheduleAt(simTime() + handoverLatency, handoverTrigger_);
}

void NRPhyUe::doHandover()
{
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
        oldAmc->detachUser(nodeId_, D2D);
    }

    if (candidateMasterId_ != 0)
    {
        LteAmc *newAmc = getAmcModule(candidateMasterId_);
        assert(newAmc != NULL);
        newAmc->attachUser(nodeId_, UL);
        newAmc->attachUser(nodeId_, DL);
        newAmc->attachUser(nodeId_, D2D);
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
    if(getParentModule()->getParentModule()->findSubmodule("NRueCollector") != -1)
    {
        binder_->moveUeCollector(nodeId_, masterId_, candidateMasterId_);
    }


    // change masterId and notify handover to the MAC layer
    MacNodeId oldMaster = masterId_;
    masterId_ = candidateMasterId_;
    mac_->doHandover(candidateMasterId_);  // do MAC operations for handover
    currentMasterRssi_ = candidateMasterRssi_;
    hysteresisTh_ = updateHysteresisTh(currentMasterRssi_);

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

        // send a self-message to schedule the possible mode switch at the end of the TTI (after all UEs have performed the handover)
        cMessage* msg = new cMessage("doModeSwitchAtHandover");
        msg->setSchedulingPriority(10);
        scheduleAt(NOW, msg);
    }

    // update DL feedback generator
    LteDlFeedbackGenerator* fbGen;
    if (!isNr_)
        fbGen = check_and_cast<LteDlFeedbackGenerator*>(getParentModule()->getSubmodule("dlFbGen"));
    else
        fbGen = check_and_cast<LteDlFeedbackGenerator*>(getParentModule()->getSubmodule("nrDlFbGen"));
    fbGen->handleHandover(masterId_);

    // collect stat
    emit(servingCell_, (long)masterId_);

    if (masterId_ == 0)
    {
        //EV << NOW << " NRPhyUe::doHandover - UE " << nodeId_ << " detached from the network" << endl;
//        std::cout << NOW << " NRPhyUe::doHandover - UE " << nodeId_ << " detached from the network" << endl;
    }
    else
    {
        //EV << NOW << " NRPhyUe::doHandover - UE " << nodeId_ << " has completed handover to gNB " << masterId_ << "... " << endl;
//        std::cout << NOW << " NRPhyUe::doHandover - UE " << nodeId_ << " has completed handover to gNB " << masterId_ << "... " << endl;

    }

    binder_->removeUeHandoverTriggered(nodeId_);
    binder_->removeHandoverTriggered(nodeId_);

    // inform the UE's IP2Nic module to forward held packets
    IP2Nic* ip2nic =  check_and_cast<IP2Nic*>(getParentModule()->getSubmodule("ip2nic"));
    ip2nic->signalHandoverCompleteUe(isNr_);

    // inform the eNB's IP2Nic module to forward data to the target eNB
    if (oldMaster != 0 && candidateMasterId_ != 0)
    {
        IP2Nic* enbIp2nic =  check_and_cast<IP2Nic*>(getSimulation()->getModule(binder_->getOmnetId(masterId_))->getSubmodule("cellularNic")->getSubmodule("ip2nic"));
        enbIp2nic->signalHandoverCompleteTarget(nodeId_,oldMaster);
    }
}

void NRPhyUe::forceHandover(MacNodeId targetMasterNode, double targetMasterRssi)
{
    Enter_Method_Silent();
    candidateMasterId_ = targetMasterNode;
    candidateMasterRssi_ = targetMasterRssi;
    hysteresisTh_ = updateHysteresisTh(currentMasterRssi_);

    cancelEvent(handoverStarter_);  // if any
    scheduleAt(NOW, handoverStarter_);
}

void NRPhyUe::deleteOldBuffers(MacNodeId masterId)
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
    // in case of NR dual connectivity, the master can be a secondary node, hence we have to delete PDCP entities residing the node's master
    MacNodeId masterNodeId = binder_->getMasterNode(masterId);
    LtePdcpRrcEnb* masterPdcp = check_and_cast<LtePdcpRrcEnb *>(getPdcpByMacNodeId(masterNodeId));
    masterPdcp->deleteEntities(nodeId_);

    // delete queues for master at this ue
    pdcp_->deleteEntities(masterId_);
}
