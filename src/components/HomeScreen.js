import { useState } from "react";
import { Link } from "react-router-dom";
//import "./HomeScreen.css";

function HomeScreen() {
  const [room, setRoom] = useState("");
  const [username, setUsername] = useState("");

  return (
  <section class="bg-gray-50 dark:bg-gray-900">
      <div class="flex flex-col items-center justify-center px-6 py-8 mx-auto md:h-screen lg:py-0">
        <a href="#" class="flex items-center mb-6 text-2xl font-semibold text-gray-900 dark:text-white">
            <img class="w-8 h-8 mr-2" src="https://oopy.lazyrockets.com/api/v2/notion/image?src=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fdacbdef3-b26c-4e81-afb1-bc685a66e5e9%2FAIF_Logo_%ED%94%84%EB%A1%9C%ED%95%84%EC%9A%A9.png&blockId=b5b41517-eff8-4cf9-970b-ac99d1ead057&width=256" alt="logo"/>
            LIVE TRACKING    
          </a>
          <div class="w-full bg-white rounded-lg shadow dark:border md:mt-0 sm:max-w-md xl:p-0 dark:bg-gray-800 dark:border-gray-700">
            <div class="p-6 space-y-4 md:space-y-6 sm:p-8">
                <h1 class="text-xl font-bold leading-tight tracking-tight text-gray-900 md:text-2xl dark:text-white">
                    Welcome to Live Tracker
                </h1>
              <form class="space-y-4 md:space-y-6">
                
                <div class="mb-4">
                  <label class="block text-gray-700 text-sm font-bold mb-2" for="username">
                    Username
                  </label>
                  <input
                  class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" 
                  id="username" 
                  type="text" 
                  placeholder="Username"
                  value={username}
                  title="username"
                  onInput={(e) => setUsername(e.target.value)}
                />      
                </div>

                <div class="mb-4">
                  <label class="block text-gray-700 text-sm font-bold mb-2" 
                  for="room">
                    Room
                  </label>
                  <input
                  class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" 
                  id="roomname" 
                  type="text" 
                  placeholder="Room Name"
                  value={room}
                  title="room"
                  onInput={(e) => setRoom(e.target.value)}
                />      
                </div>

              <Link to={`/call/${username}/${room}`}>
                <input class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mt-4" type="submit" name="submit" value="Join Room" />
              </Link>

              <Link to={`/tracking/${username}/${room}`}>
                <input class="bg-green-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mt-4" type="submit" name="submit" value="Join Room(Track-Mode)" />
              </Link>
            </form>
              <p class="text-center text-gray-500 text-xs">
                &copy;2022 SKT AI FELLOWSHIP Precious3. All rights reserved.
              </p>
          </div>    
        </div>
      </div>
    </section>
  );
}

export default HomeScreen;

