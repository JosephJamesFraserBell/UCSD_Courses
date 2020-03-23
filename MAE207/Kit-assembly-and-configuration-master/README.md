
# Overview

For the first half of the class student teams will be working with a single leg kit. This consists of two motors and encoders, leg components, and a motor driver with power supply. The following instructions will lead you through the assembly and setup of your leg kit. 

__Disclaimer: While we are using low-voltages (24V) it is still important to follow safe laboratory practices. Always work with your partner in the lab and each power supply must be connected to the provided surge protector with on/off switch.__ 

# Assembly instructions

Follow all instructions carefully! The general order of operations should be: 

1) Hardware assembly, including: encoder mounted onto the motors, motors secured onto the 8020 aluminums, and wire connections of motors and encoders with the ODrive (board needs to be powered with 24V PS). 
2) Configuring your programming environment (using Python 3) following the 'Getting Started' [rules](https://docs.odriverobotics.com/).

3) Testing the ODrive through the [instructions](https://docs.odriverobotics.com/).

4) Connect the legs (not need for 4/10 Wed Section).

![kit component](overall_ss.PNG)



## Item list per group (2-3 ppl): 
1. BLDC motors 335KV x2 

2. Waterjet aluminum legs (distributed later) 

3. 21A~24V DC power supply x1

4. AMT 102 CUI encoder x2 (already mounted onto the motors)

5. ODrive (motor control board) x1

6. M4 bolts and drop in nuts (for 8020 T-slots) x4

7. Jumper wires  x8

8. 8020 T-slot aluminum extrusions x2

9. allen wrench for M3 socket screws x1

10. flat head screw driver x1

11. 15ft power coord (will be chopped into three pieces for the power supply and motor cables)

12. Phillips head screw driver x1


## Mechanical setup 

1. Mounting encoders (already done for you!)

![combined](combined1.png)

2. Fasten the motor on to the acrylic mounts then fasten the whole thing onto the 8020s.


<img src="https://github.com/MAE207-BioinspiredMobileRobotics/Kit-assembly-and-configuration/blob/master/2nd.PNG" height="65%" width="65%">

## Electrical connections:

1. The power supply is connected to the wall using the provided 3-wire cord. Cut the 3-wire coord (__while unplugged!__) approximately 2 feet from the plug end. Strip the wires and connect the color codes to the power supply terminals (see the table below). Use two 12 gauge wires to connect the power supply to the ODrive. 

_Note_: (on the power supply)  

| Power supply terminal | Connected to |
|-----------|:--------|
| G (the ground symbol) | Green wire | 
| N | Black wire | 
| L | White wire | 
| +V | +V (ODrive) |
| -V | -V (ODrive) |
                                 

2. Motors (3 wires on each) connected to both M0 and M1 port. There are three wires coming out of the motor, the middle wire __MUST__ be connected to the middle terminal on the ODriver but the 1st and 3rd wires can be connected to either ODrive end (this will only affect the default rotating direction of the motor).

3. Encoder must to be connected correctly using 4 jumper wires to the ports showing (M0,M1 corresponding to the M0,M1 motors as well), __or the encoder will be burned!!__
  
  | Encoder   | ODrive   |
  |-----------|:--------|
  | GND       | GND|
  | 5V        | 5V|
  | Channel A |  A|
  | Channel B | B|
  
4. An overall hardware connection is shown in this picture. (A shunt resistor needs to be soldered with 2 wires and be attached to the ODrive)

![elec](elec.png)

5. Flashing the ODrive using the programmer such that when the ODrive starts, you will hear the "beep" sound and the motors start to calibrate themselves.

The Electrical connections should look like the image above. In (1) we show the power supply connections (__make sure to replace the cover and the plastic terminal covers before using!__). In (2) we show the motor & encoder connections to the ODrive and the power resistor. (3) Shows the encoders. (4) shows the location of the programmer connections 

Now we finished all the electronic connection parts  (o_o)!

## Software configuration:

All instructions about software configuration can be found on the [website](https://docs.odriverobotics.com/). This includes, setting up progamming environments for both Windows and Mac users, commands and different control modes. 

## Testing communication and control of the motors without the legs

Following the [instructions](https://docs.odriverobotics.com/) for simple commands of motor control (e.g. moving to a position or running at a constant speed). 

    a. Change the motor magent pole pairs to 7 (if it is not 7).
    
    b. Verify the encoder CPR is 2048*4.



## Leg assembly

<img src="https://github.com/MAE207-BioinspiredMobileRobotics/Kit-assembly-and-configuration/blob/master/final_version.PNG" height="80%" width="80%">

In this part we will assemble the leg for simple practics of the motor-leg system.

The hardware you need are:

1. M3x16 screws x4
2. M3x20 screws x4
3. laser cut acrylic spacer x1 (to level up one leg for better alignment)
4. waterjet aluminum leg pcs x4 (with 5x10 bearings already loctite in)
5. M4 shoudler bolts x3
6. shims (0.3/0.5mm) the number depends


#### 2 steps for building the motor-leg system:

Step1: assembling the upper part of the legs.
Notice that we have 4 pcs of leg components forming a closed loop geometry (parallel leg). __Therefore, offseting one leg (using a piece of acrylic spacer) is crucial for overall alignment, which significantly reduces the friction.__ 

<img src="https://github.com/MAE207-BioinspiredMobileRobotics/Kit-assembly-and-configuration/blob/master/step1.PNG" height="70%" width="70%">

Step2: Connecting the lower part of the legs using shoulder bolts.

The shoulder bolt has a non-threaded clean cylinder part that can be used as a shaft to match with the bearing. However, the non-threaded part is a little longer than the thickness of the alumium legs we have. Therefore, using shims to raise the bolts up so that the assembled legs can be flash on surface is helpful. However, inserting too many shims will make your leg system frictional, for any 2 legs being pushed against each other (by the bolt) will cause frictions between the surface when rotating. We just want a very little tiny gap between the 2 legs so that they can rotate freely.

<img src="https://github.com/MAE207-BioinspiredMobileRobotics/Kit-assembly-and-configuration/blob/master/step2.PNG" height="70%" width="70%">

```python

```
