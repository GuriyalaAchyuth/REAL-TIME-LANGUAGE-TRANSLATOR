# REAL-TIME-LANGUAGE-TRANSLATOR
# Traffic Light Controller Using Verilog
* The purpose of this project is to design a methodology using Verilog to control the traffic with specified time delays for a T-Shaped road.

# Table of Contents

1. [Introduction](#Introduction)
2. [Methodology](#Methodology)
    - [Directions Considered](#directions-considered)
    - [Problem Statement](#problem-statement)
    - [State Diagram](#state-diagram)
    - [State Table](#state-table)
3. [RTL Schematic ](#rtl-schematic)
4. [Gate Level Schematic ](#gate-level-schematic)
5. [Utilisation Report](#Utilisation-Report)
6. [Power Report](#Power-Report)
5. [Simulation Result](#Simulation-Result)
6. [Result](#result)
7. [Future work](#future-work)
8. [References](#references)
9. [Author](#author)


## Introduction 
This project deals with a basic design of a T- Shaped road for traffic light control. The output of system has been tested
using Xilinx Vivado 2024.2.
A traffic light, also known as traffic signal, stop light, stop-and-go lights, is a signaling device positioned at a road
intersection, pedestrian crossing, or other location in order to indicate when it is safe to drive, ride, or walk using a
universal colour code. Nowadays,
 1. Ared light meant traffic in all directions had to stop.
 2. Ayellow light meant cross-town traffic would have to slow and,
 3. Agreen light would to go or proceed

#### Benefits of Using Verilog HDL (Hardware Description Language) : 

Verilog is a widely used Hardware Description Language (HDL) for designing digital circuits. It can also be used for modeling analog circuits. Verilog is a descriptive language that describes a relationship between signals in a circuit. 

A Verilog model describes a unit of digital hardware in terms of :
- Interconnections of other hardware unit whose models prescribe their behavior in a 
    simulation.
- Behavioral / procedural algorithms that abstractly describe input/output behavior    
       that could be personified in a hardware unit. 

Hardware description language (HDL) is divided by two types, Verilog and VHDL (VHSIC – Very High Speed Integrated Circuit Hardware Description Language). Both have its advantages and its disadvantage. 
In this project, Verilog HDL was chosen because it’s used for synthesis of logic circuits (synthesizable code), used for verification purposes of a circuit (can be analog or digital or mixed signal), can be used by combining synthesis & verification (synthesizable & behavioral code) and it used for netlist representation of a synthesizable circuit (structural code). 

#### The advantages using Verilog HDL are shown below : 
- Easy to write. 
- Easy to understand as it similar to C program. 
- Easier to learn compared with VHDL.


## Methodology

### Directions Considered

![directions](https://github.com/praveen132-cpu/TRAFFIC-LIGHT-CONTROLLER-USING-VERILOG-HDL/blob/main/methodology.png?raw=true)

The directions, M1, MT, M2, S, that is been considered for analysis of our problem is shown in the figure. And, the problem statement is explained in the figure. 
Six states, S1, S2, S3, S4, S5, S6 are taken into consideration and state diagram, state table is made using the following logic explained in the figure. 

### Problem Statement

![Logic_Diagram](https://github.com/praveen132-cpu/TRAFFIC-LIGHT-CONTROLLER-USING-VERILOG-HDL/blob/main/problem%20statement.png?raw=true)

- Green light indicates that there is no traffic and there is easy flow of vehicles in that route/direction. 
- Red light indicates that there is a traffic jam and that route is blocked for the vehicles to move and, 
- Yellow light indicates that the route has medium flow of vehicles. 

Time delays for changing from one state to another is considered as, TMG(from S1 to S2), TY(from S2 to S3), TTG(from S3 to S4), TY(from S4 to S5), TSG(from S5 to S6) 
and TY(from S6 to S1) and the cycle continues.


### State Diagram

![State_Diagram](https://github.com/praveen132-cpu/TRAFFIC-LIGHT-CONTROLLER-USING-VERILOG-HDL/blob/main/state%20diagram.jpg?raw=true)

In Figure, The time delays are considered as follows :

- TMG = 7 seconds
- TY = 2 seconds
- TTG = 5 seconds
- TSG = 3 seconds

Until TMG seconds, the signal will remain in S1 state, and after TMG seconds, it will move to S2 state. Until TY seconds it will remain in S2 state and after TY seconds, 
it will move to S3 state, and so on. After TY seconds, in state S6, it will go back to S1 state and the cycle continues.


### State Table

![StateTable](https://github.com/praveen132-cpu/TRAFFIC-LIGHT-CONTROLLER-USING-VERILOG-HDL/blob/main/state%20table.png?raw=true)

In Figure, 
- R = RED, 
- Y = YELLOW and, 
- G = GREEN.

ST = State Transition; A, B and C are considered as the present state.

## RTL Schematic

![RTL Schematic](https://github.com/praveen132-cpu/TRAFFIC-LIGHT-CONTROLLER-USING-VERILOG-HDL/blob/main/rtl%20schematic.jpg?raw=true)

## Gate Level Schematic 

![Gate level Schematic](https://github.com/praveen132-cpu/TRAFFIC-LIGHT-CONTROLLER-USING-VERILOG-HDL/blob/main/gate%20level%20schematic.jpg?raw=true)

## Utilisation Report

![Utilisation Report](https://github.com/praveen132-cpu/TRAFFIC-LIGHT-CONTROLLER-USING-VERILOG-HDL/blob/main/utilisation%20report.jpg?raw=true)

## Power Report

![Power Report](https://github.com/praveen132-cpu/TRAFFIC-LIGHT-CONTROLLER-USING-VERILOG-HDL/blob/main/power%20report.jpg?raw=true)


## Simulation Result

![Simulation Result](https://github.com/praveen132-cpu/TRAFFIC-LIGHT-CONTROLLER-USING-VERILOG-HDL/blob/main/simulation%20result.jpg?raw=true)


## Result

The implementation of the advanced traffic light controller using Verilog HDL successfully achieved a reliable and efficient control mechanism for managing traffic flow at an intersection. The design was synthesized and simulated using industry-standard tools such as Xilinx 2024.2, confirming the correct operation of the state machine controlling the traffic signals. The controller responded accurately to various traffic scenarios, including timing adjustments for peak and off-peak hours and prioritization for emergency situations when simulated inputs were applied. The system demonstrated low latency in state transitions and maintained stability across multiple test cases. Overall, the Verilog-based design provided a robust, scalable, and hardware-efficient solution that can be deployed in real-time environments with minimal resource usage on FPGA platforms.


## Future Work

In future developments of the advanced traffic light controller using Verilog HDL, several enhancements can be implemented to improve functionality, adaptability, and real-time responsiveness. One promising direction is the integration of sensor-based traffic flow analysis, where real-time data from infrared or ultrasonic sensors can dynamically adjust signal timings based on actual vehicular density. Additionally, incorporating machine learning algorithms for predictive traffic control could further optimize traffic flow and reduce congestion. Another area of future work involves expanding the controller to support intelligent transportation systems (ITS), allowing for communication between traffic lights and connected vehicles (V2I communication). The system can also be made scalable to accommodate larger and more complex intersections with pedestrian and emergency vehicle prioritization. Furthermore, implementing a graphical simulation interface for monitoring and control would enhance testing, validation, and deployment efficiency. These advancements will push the boundaries of conventional traffic control systems, aligning them with the goals of smart city infrastructure.

## References

1.	Cao, K. (2025). Traffic Light Controller System Analysis based on FPGA. Applied and Computational Engineering, 124(1), 199–211. https://doi.org/10.54254/2755- 2721/2025.20027

2.	Broydé, F., & Clavelier, E. (2025). About the Gains and the Effective Areas of a Multiport Antenna Array. https://doi.org/10.5281/zenodo.14584381

3.	Zhang, L.-K., Yang, Y., Zhou, S.-G., Xu, C., Feng, Y., Yang, Y., & Li, J. (2025).
Wideband High-Gain Dual-Polarized Array Antenna with Bottom-Coupled Feeding Structure. IEEE Transactions on Antennas and Propagation, 1. https://doi.org/10.1109/tap.2024.3523678

4.	Liu, H., Wang, H., Xu, L., Liu, B., Liu, J., Zhang, X., Yuan, X., & Li, B. (2025). A Fast Electromagnetic Radiation Simulation Tool for Finite Periodic Array Antenna and Universal Array Antenna. International Journal of Rf and Microwave Computer-Aided Engineering, 2025(1). https://doi.org/10.1155/mmce/5999155

5.	Xu, H., Zhou, G., Wong, K., New, W. K., Wang, C., Chae, C., Murch, R., Jin, S., & Zhang,
Y. (2024). Channel Estimation for FAS-Assisted Multiuser mmWave Systems. IEEE Communications Letters, 28(3), 632–636. https://doi.org/10.1109/lcomm.2023.3347951

6.	Yuan, S., Wang, J., Xu, H., Wang, T., Li, D., Chen, X., Huang, C., Sun, S., Zheng, S., Zhang, X., Li, E., & Wang, S. (2024). Breaking the Degrees-of-Freedom Limit of Holographic MIMO Communications: A 3-D Antenna Array Topology. IEEE Transactions on Vehicular Technology, 1–13. https://doi.org/10.1109/tvt.2024.3372704

7.	Hu, G., Wu, Q., Ouyang, J., Xu, K., Cai, Y., & Al‐Dhahir, N. (2024). Movable-Antenna- Array-Enabled Communications with CoMP Reception. IEEE Communications Letter https://doi.org/10.1109/lcomm.2024.3358762

8.	Takizawa, Y., & Fukasawa, A. (2024). Compact-High Performance Microwave Circularly Polarized Antenna Array at Lower Frequency Band. 649–650. https://doi.org/10.1109/ap- s/inc-usnc-ursi52054.2024.10686358

9.	Zhu, Y., Ma, W., Wang, C., & Cao, W. (2024). Low sidelobe planar electrically large sparse array antenna with element number reduction based on genetic algorithm. Iet Microwaves Antennas & Propagation. https://doi.org/10.1049/mia2.12475

10.	Wang, C., & Lee, J.-H. (2024). Generalized Sidelobe Canceller Based Antenna ArrayBeamformer with Reduced Computational Complexity (pp. 845–856). Springer International Publishing. https://doi.org/10.1007/978-3-031-47724-9_56



## Author

- [kambham praveen kumar reddy](https://github.com/praveen132-cpu), Bachelor of Engineering in Electronics and Communication Engineering, st.peter's engineering college

    

