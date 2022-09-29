import { Routes, Route } from "react-router-dom";
import CallScreen from "./CallScreen";
import HomeScreen from "./HomeScreen";
import TrackScreen from "./TrackScreen";

function RouteList() {
  return (
    <Routes>
      <Route path="/" element={<HomeScreen />} />
      <Route path="/call/:username/:room" element={<CallScreen />} />
      <Route path="/tracking/:username/:room" element={<TrackScreen />} />
    </Routes>
  );
}

export default RouteList;
