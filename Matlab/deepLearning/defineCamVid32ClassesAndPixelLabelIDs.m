function [classes, labelIDs] = defineCamVid32ClassesAndPixelLabelIDs()
% The defineCamVid32ClassesAndPixelLabelIDs function returns the class
% names and the corresponding label IDs for each class in the CamVid
% dataset.
%
% CamVid pixel label IDs are a cell array of M-by-3 matrices,
% where M is the number of labels of each class. The original CamVid class
% names are listed alongside each RGB value.

classes = [
    "Animal"
    "Archway"
    "Bicyclist"
    "Bridge"
    "Building"
    "Car"
    "CartLuggagePram"
    "Child"
    "Column_Pole"
    "Fence"
    "LaneMkgsDriv"
    "LaneMkgsNonDriv"
    "Misc_Text"
    "MotorcycleScooter"
    "OtherMoving"
    "ParkingBlock"
    "Pedestrian"
    "Road"
    "RoadShoulder"
    "Sidewalk"
    "SignSymbol"
    "Sky"
    "SUVPickupTruck"
    "TrafficCone"
    "TrafficLight"
    "Train"
    "Tree"
    "Truck_Bus"
    "Tunnel"
    "VegetationMisc"
    "Void"
    "Wall"
    ];

labelIDs = { ...
    % Animal
    [ 
    64 128 64;   
    ] 
    
    % Archway
    [
    192 0 128;
    ]

    % Biyclist
    [
    0 128 192;
    ]

    % Bridge
    [
    0 128 64;
    ]

    % Building
    [
    128 0 0;
    ]

    % Car
    [
    64 0 128;
    ]

    % CartLuggagePram
    [
    64 0 192;
    ]
    
    % Child
    [
    192 128 64;
    ]
    
    % Column Pole
    [
    192 192 128;
    ]
    
    % Fence
    [
    64 64 128;
    ]

    % LaneMkgsDriv
    [
    128 0 192;
    ]

    % LaneMkgsNonDriv
    [
    192 0 64;
    ]

    % Misc Text
    [
    128 128 64;
    ]

    % MotorcycleScooter
    [
    192 0 192;
    ]

    % OtherMoving
    [
    128 64 64;
    ]

    % ParkingBlock
    [
    64 192 128;
    ]

    % Pedestrian
    [
    64 64 0;
    ]

    % Road
    [
    128 64 128;
    ]

    % RoadShoulder
    [
    128 128 192;
    ]

    % Sidewalk
    [
    0 0 192;
    ]

    % SignSymbol
    [
    192 128 128;
    ]

    % Sky
    [
    128 128 128;
    ]

    % SUVPickupTruck
    [
    64 128 192;
    ]

    % TrafficCone
    [
    0 0 64;
    ]

    % TrafficLight
    [
    0 64 64;
    ]

    % Train
    [
    192 64 128;
    ]

    % Tree
    [
    128 128 0;
    ]

    % Truck Bus
    [
    192 128 192;
    ]

    % Tunnel
    [
    64 0 64;
    ]

    % VegetationMisc
    [
    192 192 0;
    ]

    % Void
    [
    0 0 0;
    ]

    % Wall
    [
    64 192 0;
    ]
   };
    
end