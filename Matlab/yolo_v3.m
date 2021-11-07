
% Welcome to Yolo v3 training for face mask detection

% Openning images in datastore
images = imageDatastore('data/train'); % make sure to add full path for this one
imagesTest = imageDatastore('data/test'); % the path should be changed based on your location path
imagesValidation = imageDatastore('data/validation'); % the path should be changed based on your location path

% Labeling images
imageLabeler(images); %or you can open trainingLabels.mat gTruth
imageLabeler(imagesTest); %or you can open testLabels.mat gTruth
imageLabeler(imagesValidation); %or you can open validLabels.mat gTruth
% Prepairing training data
%trainingLabels = load('trainingLabels.mat');
% getting box labels
trainingLabels = sortrows(trainingLabels,1);
blds = boxLabelDatastore(trainingLabels(:,2:3));
% combinig data
trainingCds = combine(images,blds);
% validate correctness of images
validateInputData(trainingCds);
% Augment data for the model
augmentedTrainingData = transform(trainingCds, @augmentData);
% Visualize the augmented images.
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1,1}, 'Rectangle', data{1,2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData, 'BorderSize', 10);


% preparing input size and anchor boxes
networkInputSize = [227 227 3];
rng(0);
trainingDataForEstimation = transform(trainingCds, @(data)preprocessData(data, networkInputSize));
numAnchors = 10;
[anchors, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

area = anchors(:, 1).*anchors(:, 2);
[~, idx] = sort(area, 'descend');
anchors = anchors(idx, :);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    };

% choosing base network
baseNetwork = squeezenet;
classNames = trainingLabels.Properties.VariableNames(2:end);
% selecting Yolo v3 object detector
yolov3Detector = yolov3ObjectDetector(baseNetwork, classNames, anchorBoxes, 'DetectionNetworkSource', {'fire9-concat', 'fire5-concat'});
% preprocessing data
preprocessedTrainingData = transform(augmentedTrainingData, @(data)preprocess(yolov3Detector, data));
%showing sample
data = read(preprocessedTrainingData);
I = data{1,1};
bbox = data{1,2};
annotatedImage = insertShape(I, 'Rectangle', bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage);
reset(preprocessedTrainingData);

% optimizing model parameters
numEpochs = 100;
miniBatchSize = 8;
learningRate = 0.001;
warmupPeriod = 1000;
l2Regularization = 0.0005;
penaltyThreshold = 0.5;
velocity = [];
% using parallel computing for training model
if canUseParallelPool
   dispatchInBackground = true;
else
   dispatchInBackground = false;
end
% mini batch ques for training
mbqTrain = minibatchqueue(preprocessedTrainingData, 2,...
        "MiniBatchSize", miniBatchSize,...
        "MiniBatchFcn", @(images, boxes, labels) createBatchData(images, boxes, labels, classNames), ...
        "MiniBatchFormat", ["SSCB", ""],...
        "DispatchInBackground", dispatchInBackground,...
        "OutputCast", ["", "double"]);
    
 
%train

% if you don't want to train, load pretrained model trained2.mat
% trained2.mat is the final model and trained1.mat is the initial trained
% model
% obviously trained2.mat has better precision rate


% Create subplots for the learning rate and mini-batch loss.
fig = figure;
[lossPlotter, learningRatePlotter] = configureTrainingProgressPlotter(fig);

iteration = 0;
% Custom training loop.
for epoch = 1:numEpochs

    reset(mbqTrain);
    shuffle(mbqTrain);

    while(hasdata(mbqTrain))
        iteration = iteration + 1;

        [XTrain, YTrain] = next(mbqTrain);

        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients function.
        [gradients, state, lossInfo] = dlfeval(@modelGradients, yolov3Detector, XTrain, YTrain, penaltyThreshold);

        % Apply L2 regularization.
        gradients = dlupdate(@(g,w) g + l2Regularization*w, gradients, yolov3Detector.Learnables);

        % Determine the current learning rate value.
        currentLR = piecewiseLearningRateWithWarmup(iteration, epoch, learningRate, warmupPeriod, numEpochs);

        % Update the detector learnable parameters using the SGDM optimizer.
        [yolov3Detector.Learnables, velocity] = sgdmupdate(yolov3Detector.Learnables, gradients, velocity, currentLR);

        % Update the state parameters of dlnetwork.
        yolov3Detector.State = state;

        % Display progress.
        displayLossInfo(epoch, iteration, currentLR, lossInfo);  

        % Update training plot with new points.
        updatePlots(lossPlotter, learningRatePlotter, iteration, currentLR, lossInfo.totalLoss);
    end        
end

%save trained model
trained2 = yolov3Detector;
save trained2;

%testData 
%precision rate for this one is 0.4 
imagesTest = imageDatastore('data/test'); % the path should be changed based on your location path
bldsTest = boxLabelDatastore(testLabels(:,2:3));
testCds = combine(imagesTest,bldsTest);
validateInputData(testCds);
results = detect(yolov3Detector,testCds,'threshold', 0.5);
% Evaluate the object detector using Average Precision metric.
[ap,recall,precision] = evaluateDetectionPrecision(results,testCds , 0.5);
% plotting
figure;
plot(recall{1},precision{1});
xlabel('Recall')
ylabel('Precision')
grid on
tempM = sprintf(' = %.1f', ap(1));
title(sprintf(['Average Precision for ', testLabels.Properties.VariableNames{2}, tempM]));

%testMyPic
I = imread('testPicY.jpg');
[bboxes,scores,labels] = detect(yolov3Detector,I);
% Display the detections on image.
%I = insertObjectAnnotation(I,'rectangle',bboxes,labels);
I = insertObjectAnnotation(I, 'rectangle', bboxes, [string(labels)+ " : "+string(scores)], 'Color', 'magenta', ...
            'Fontsize', 30, 'linewidth', 10, 'textboxopacity', 0.4);
figure
imshow(I);

%validating data for comparison to other group members trained models 
%percision rate for this one is 0.6
bldsValidation = boxLabelDatastore(ValidationLabels2(:,2:3));
ValidationCds = combine(imagesValidation,bldsValidation);
validateInputData(ValidationCds);
results = detect(yolov3Detector,ValidationCds,'threshold', 0.5);
% Evaluate the object detector using Average Precision metric.
[ap,recall,precision] = evaluateDetectionPrecision(results,ValidationCds , 0.5);
% plotting
figure;
plot(recall{1},precision{1});
xlabel('Recall')
ylabel('Precision')
grid on
tempM = sprintf(' = %.1f', ap(1));
title(sprintf(['Average Precision for ', testLabels.Properties.VariableNames{2}, tempM]));
