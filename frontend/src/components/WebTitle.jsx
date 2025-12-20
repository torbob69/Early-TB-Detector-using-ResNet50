import { Typewriter } from "react-simple-typewriter";

function WebTitle(){
    return(
        <h1 className="text-4xl md:text-3xl lg:text-6xl font-extralight text-white mb-4 md:mb-2 tracking-tight lg:text-left">
              <Typewriter
                words={["Early TB Detector.", "Up to 99% Accuracy."]}
                loop={true}
                typeSpeed={100}
                deleteSpeed={100}
                delaySpeed={5000}
                cursor={true}
              />
            </h1>
    );
}

export default WebTitle;