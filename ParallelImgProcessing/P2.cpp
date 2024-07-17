#include <mpi.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes and the rank of the current process
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

     string imagePath, outputPath;
     double start_time, end_time;
     Mat image;
     int rows, cols;
     int blurRadius;
     int threshold1, threshold2;
     int neighborhood_size;
     int thresholdValue;
     int choice;
    if (world_rank == 0) {
        // Display the menu on the root process
        cout << "Welcome to Parallel Image Processing with MPI" << endl;
        cout << "Please choose an image processing operation:" << endl;
        cout << "01- Gaussian Blur" << endl;
        cout << "02- Local Thresholding" << endl;
        cout << "03- Global Thresholding" << endl;
        cout << "04- Median" << endl;
        cout << "05- Histogram Equalization" << endl;
        cout << "06- Edge Detection" << endl;
        cout << "07- Color Space Conversion" << endl;
        cout << "Enter your choice (1-7): ";
        cin >> choice;
        if (choice == 1) {
            cout << "You have selected Gaussian Blur. " << endl;
            cout << "Please enter the filename of the input image (e.g., input.jpg):  ";
            cin >> imagePath;
            image = imread(imagePath);
            // Check if the image is loaded successfully
            if (image.empty()) {
                cerr << "Error: Unable to load the image." << endl;
                MPI_Finalize();
                return 1;
            }
            rows = image.rows;
            cols = image.cols;
            cout << "Please enter the filename for the output blurred image (e.g., output.jpg): ";
            cin >> outputPath;
            cout << "Please enter the blur radius (e.g., odd):";
            cin >> blurRadius;
        }
        else if (choice == 2) {
            cout << "You have selected Local Thresholding. " << endl;
            cout << "Please enter the filename of the input image (e.g., input.jpg):  ";
            cin >> imagePath;
            image = imread(imagePath);
            // Check if the image is loaded successfully
            if (image.empty()) {
                cerr << "Error: Unable to load the image." << endl;
                MPI_Finalize();
                return 1;
            }
            rows = image.rows;
            cols = image.cols;
            cout << "Please enter the filename for the output threshold image (e.g., output.jpg): ";
            cin >> outputPath;
            cout << "Enter the neighborhood size (e.g., 3 for a 3x3 window): ";
            cin >> neighborhood_size;
        }
        else if (choice == 3) {
            cout << "You have selected Global Thresholding. " << endl;
            cout << "Please enter the filename of the input image (e.g., input.jpg):  ";
            cin >> imagePath;
            image = imread(imagePath);
            // Check if the image is loaded successfully
            if (image.empty()) {
                cerr << "Error: Unable to load the image." << endl;
                MPI_Finalize();
                return 1;
            }
            rows = image.rows;
            cols = image.cols;
            cout << "Please enter the filename for the output threshold image (e.g., output.jpg): ";
            cin >> outputPath;
            cout << "Enter the Threshold Value (0-255): ";
            cin >> thresholdValue;
        }
        else if (choice == 4) {
            cout << "You have selected Median. " << endl;
            cout << "Please enter the filename of the input image (e.g., input.jpg):  ";
            cin >> imagePath;
            image = imread(imagePath);
            // Check if the image is loaded successfully
            if (image.empty()) {
                cerr << "Error: Unable to load the image." << endl;
                MPI_Finalize();
                return 1;
            }
            rows = image.rows;
            cols = image.cols;
            cout << "Please enter the filename for the output image (e.g., output.jpg): ";
            cin >> outputPath;
        }
        else if (choice == 5) {
            cout << "You have selected Histogram Equalization. " << endl;
            cout << "Please enter the filename of the input image (e.g., input.jpg):  ";
            cin >> imagePath;
            image = imread(imagePath);
            // Check if the image is loaded successfully
            if (image.empty()) {
                cerr << "Error: Unable to load the image." << endl;
                MPI_Finalize();
                return 1;
            }
             rows = image.rows;
             cols = image.cols;
            cout << "Please enter the filename for the output equalized image (e.g., output.jpg): ";
            cin >> outputPath;
        }
        else if (choice == 6) {
            cout << "You have selected Edge Detection (Canny). " << endl;
            cout << "Please enter the filename of the input image (e.g., input.jpg):  ";
            cin >> imagePath;
             image = imread(imagePath);
            // Check if the image is loaded successfully
            if (image.empty()) {
                cerr << "Error: Unable to load the image." << endl;
                MPI_Finalize();
                return 1;
            }
             rows = image.rows;
             cols = image.cols;
            cout << "Please enter the filename for the output edge detection image (e.g., output.jpg): ";
            cin >> outputPath;
            cout << "Please enter the Threshold1 value (0-255) : ";
            cin >> threshold1;
            cout << "Please enter the Threshold2 value (0-255) : ";
            cin >> threshold2;
        }
        else if (choice == 7) {
             cout << "You have selected Color Space Conversion. " << endl;
             cout << "Please enter the filename of the input image (e.g., input.jpg):  ";
             cin >> imagePath;
             image = imread(imagePath);
             // Check if the image is loaded successfully
            if (image.empty()) {
                cerr << "Error: Unable to load the image." << endl;
                MPI_Finalize();
                return 1;
            }
             rows = image.rows;
             cols = image.cols;
            cout << "Please enter the filename for the output color conversion images (e.g., output.jpg): ";
            cin >> outputPath;
        }
    }

    // Broadcast the choice to all processes
    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast image path, output path, and blur radius to all processes
    int imagePathSize, outputPathSize;
    if (world_rank == 0) {
        imagePathSize = imagePath.size() + 1; // Including null terminator
        outputPathSize = outputPath.size() + 1; // Including null terminator
    }
    MPI_Bcast(&imagePathSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&outputPathSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast image path
    char* imagePathBuffer = new char[imagePathSize];
    if (world_rank == 0) {
        imagePath.copy(imagePathBuffer, imagePath.size());
        imagePathBuffer[imagePath.size()] = '\0'; // Null terminator
    }
    MPI_Bcast(imagePathBuffer, imagePathSize, MPI_CHAR, 0, MPI_COMM_WORLD);
    imagePath = string(imagePathBuffer); // Convert char array back to string

    // Broadcast output path
    char* outputPathBuffer = new char[outputPathSize];
    if (world_rank == 0) {
        outputPath.copy(outputPathBuffer, outputPath.size());
        outputPathBuffer[outputPath.size()] = '\0'; // Null terminator
    }
    MPI_Bcast(outputPathBuffer, outputPathSize, MPI_CHAR, 0, MPI_COMM_WORLD);
    outputPath = string(outputPathBuffer); // Convert char array back to string

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate local portion of the image
    int rows_per_process = rows / world_size;
    int start_row = world_rank * rows_per_process;
    int end_row = (world_rank + 1) * rows_per_process;
    if (world_rank == world_size - 1) {
        end_row = rows;
    }
     int local_rows = end_row - start_row;

     // Allocate memory for local image portion
     Mat local_image(local_rows, cols, CV_8UC3);

     // Broadcast blur radius
     MPI_Bcast(&blurRadius, 1, MPI_INT, 0, MPI_COMM_WORLD);

     // Broadcast threshold values to all processes
     MPI_Bcast(&threshold1, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&threshold2, 1, MPI_INT, 0, MPI_COMM_WORLD);

     // Broadcast neighborhood_size value to all processes
     MPI_Bcast(&neighborhood_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

     // Broadcast Threshold value to all processes
     MPI_Bcast(&thresholdValue, 1, MPI_INT, 0, MPI_COMM_WORLD);


    // Process based on the user's choice
    switch (choice) {
    case 1:
    {
       
        // to start time
        start_time = MPI_Wtime();

        // Debugging output for checking 
       /*cout << "Rank " << world_rank << ": Image rows = " << local_image.rows << ", cols = " << local_image.cols << endl;*/

        // Convert local image to linear array
        int total_pixels = local_image.rows * local_image.cols;
        int bytes_per_pixel = local_image.elemSize();
        int total_bytes = total_pixels * bytes_per_pixel;
        uchar* local_data = new uchar[total_bytes];
        memcpy(local_data, local_image.data, total_bytes);

        // Scatter processes
        uchar* recv_data = new uchar[total_bytes];
        MPI_Scatter(image.data, total_bytes, MPI_CHAR, recv_data, total_bytes, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Construct the local image from received data
        Mat received_image(local_image.rows, local_image.cols, local_image.type(), recv_data);

        GaussianBlur(received_image, received_image, Size(blurRadius, blurRadius), 0, 0);

        // Gather the processed images on the master 
        Mat globalBlurredImage;
        if (world_rank == 0) {
            globalBlurredImage.create(image.size(), image.type());
        }
        int* recv_counts = new int[world_size];
        MPI_Gather(&total_bytes, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int* displacements = new int[world_size];
        if (world_rank == 0) {
            displacements[0] = 0;
            for (int i = 1; i < world_size; ++i) {
                displacements[i] = displacements[i - 1] + recv_counts[i - 1];
            }
        }
        MPI_Gatherv(received_image.data, total_bytes, MPI_CHAR, globalBlurredImage.data, recv_counts, displacements, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Print out the number of rows processed by each process for checking 
        /*int* all_rows_processed = new int[world_size];
        MPI_Gather(&local_image.rows, 1, MPI_INT, all_rows_processed, 1, MPI_INT, 0, MPI_COMM_WORLD);*/
        /* if (world_rank == 0) {
             cout << "Rows processed by each process: ";
             for (int i = 0; i < world_size; ++i) {
              cout << all_rows_processed[i] << " ";
          }
          cout << endl;
        }*/

        // Display the output image on the master 
        if (world_rank == 0) {
            cout << "Processing image " << imagePath << " with Gaussian Blur..." << endl;

            imwrite(outputPath, globalBlurredImage);
            cout << "Blurred image saved as " << outputPath << endl;
            end_time = MPI_Wtime();

            double elapsed_time = end_time - start_time;
            cout << "Gaussian Blur operation completed successfully in " << elapsed_time << " seconds. " << endl;
            cout << "Thank you for using Our Parallel Image Processing with MPI." << endl;
        }
    }
       break;
    case 2:
    {
        start_time = MPI_Wtime();

        // Scatter 
        MPI_Scatter(image.data + start_row * cols, local_rows * cols, MPI_CHAR,
            local_image.data, local_rows * cols, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Perform local thresholding on the local portion of the image
        Mat local_thresholded = local_image.clone();
        for (int i = 0; i < local_rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                // Define the local neighborhood
                int sum = 0;
                int count = 0;
                for (int x = i - neighborhood_size / 2; x <= i + neighborhood_size / 2; ++x) {
                    for (int y = j - neighborhood_size / 2; y <= j + neighborhood_size / 2; ++y) {
                        if (x >= 0 && x < local_rows && y >= 0 && y < cols) {
                            sum += local_image.at<uchar>(x, y);
                            count++;
                        }
                    }
                }
                // Compute the local threshold as the mean of the neighborhood
                int local_threshold = sum / count;

                if (local_image.at<uchar>(i, j) < local_threshold) {
                    local_thresholded.at<uchar>(i, j) = 0;     // Below threshold set to black
                }
                else {
                    local_thresholded.at<uchar>(i, j) = 255;    // Above threshold set to white
                }
            }
        }

        // Debug: Output the local thresholded image for each process
        for (int p = 0; p < world_size; ++p) {
            if (world_rank == p) {
                cout << "Process " << world_rank << " local thresholded image:" << endl;
                cout << local_thresholded << endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Gather local thresholded images on the master 
        Mat global_thresholded;
        if (world_rank == 0) {
            global_thresholded = Mat::zeros(rows, cols, CV_8U);
        }

        // Allocate memory for receive buffer and calculate displacements
        int* recv_counts = new int[world_size];
        int* displacements = new int[world_size];
        int total_elements = rows * cols;
        for (int i = 0; i < world_size; ++i) {
            recv_counts[i] = (i == world_size - 1) ? total_elements - i * rows_per_process * cols : rows_per_process * cols;
            displacements[i] = i * rows_per_process * cols;
        }

        // Gather local thresholded images with MPI_Gatherv
        MPI_Gatherv(local_thresholded.data, local_rows * cols, MPI_CHAR,
            global_thresholded.data, recv_counts, displacements, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Debug: Output the gathered global thresholded image on the master
        if (world_rank == 0) {
            cout << "Gathered global thresholded image on master:" << endl;
            cout << global_thresholded << endl;
        }

        // Save final global thresholded image on the master
        if (world_rank == 0) {
            cout << "Processing Image " << imagePath << " with Local Thresholding..." << endl;
            cv::imwrite(outputPath, global_thresholded);
            cout << "Local Threshold Image saved as " << outputPath << endl;

            end_time = MPI_Wtime();
            double elapsed_time = end_time - start_time;
            cout << "Local Thresholding operation completed successfully in " << elapsed_time << " seconds. " << endl;
            cout << "Thank you for using Our Parallel Image Processing with MPI." << endl;
        }

        // Clean up memory
        delete[] recv_counts;
        delete[] displacements;


    }
    break;
    case 3:
    {
        start_time = MPI_Wtime();

        // Scatter image 
        MPI_Scatter(image.data + start_row * cols, local_rows * cols, MPI_CHAR,
            local_image.data, local_rows * cols, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Apply global thresholding to the local portion of the image
        threshold(local_image, local_image, thresholdValue, 255, THRESH_BINARY);

        // Gather local thresholded images on the master 
        Mat global_thresholded;
        if (world_rank == 0) {
            global_thresholded = Mat::zeros(rows, cols, CV_8U);
        }

        // Allocate memory for receive buffer and calculate displacements
        int* recv_counts = new int[world_size];
        int* displacements = new int[world_size];
        int total_elements = rows * cols;
        for (int i = 0; i < world_size; ++i) {
            recv_counts[i] = (i == world_size - 1) ? total_elements - i * rows_per_process * cols : rows_per_process * cols;
            displacements[i] = i * rows_per_process * cols;
        }

        // Gather local thresholded images with MPI_Gatherv
        MPI_Gatherv(local_image.data, local_rows * cols, MPI_CHAR,
            global_thresholded.data, recv_counts, displacements, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Save or process the final global thresholded image on the master process
        if (world_rank == 0) {
            cout << "Processing Image " << imagePath << " with Global Thresholding..." << endl;
            cv::imwrite(outputPath, global_thresholded);
            cout << "Global Threshold Image saved as " << outputPath << endl;

            end_time = MPI_Wtime();
            double elapsed_time = end_time - start_time;
            cout << "Global Thresholding operation completed successfully in " << elapsed_time << " seconds. " << endl;
            cout << "Thank you for using Our Parallel Image Processing with MPI." << endl;
        }

        // Clean up memory
        delete[] recv_counts;
        delete[] displacements;
    }
    break;
    case 4:
    {
        start_time = MPI_Wtime();
        cv::Mat originalImage = cv::imread(imagePath);

        // Split the image into its RGB channels
        vector<cv::Mat> originalChannels(3);
        cv::split(originalImage, originalChannels);

        int channelCount = originalChannels.size();
        int pixelCount = originalChannels[0].total();
        int l = pixelCount / world_size;

        vector<int> medianValues(channelCount, 0);

        for (int channel = 0; channel < channelCount; ++channel) {
            // Calculate median for each channel separately
            uchar* byteArray = originalChannels[channel].data;

            // Scatter data to different processes
            uchar* recArray = new uchar[l];
            MPI_Scatter(byteArray, l, MPI_CHAR, recArray, l, MPI_CHAR, 0, MPI_COMM_WORLD);

            // Sort the received data to find the median
            sort(recArray, recArray + l);

            // Find the median
            medianValues[channel] = l % 2 == 0 ? (recArray[l / 2] + recArray[l / 2 - 1]) / 2 : recArray[l / 2];

            // Free allocated memory
            delete[] recArray;
        }

        // Gather median values from all processes
        vector<int> allMedians(channelCount * world_size);
        MPI_Gather(&medianValues[0], channelCount, MPI_INT, &allMedians[0], channelCount, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            // Calculate the final median values
            vector<int> finalMedians(channelCount, 0);
            for (int channel = 0; channel < channelCount; ++channel) {
                vector<int> channelMedians;
                for (int i = 0; i < world_size; ++i) {
                    channelMedians.push_back(allMedians[i * channelCount + channel]);
                }
                sort(channelMedians.begin(), channelMedians.end());
                finalMedians[channel] = channelMedians[world_size / 2];
            }

            cout << "Final Median Values: ";
            for (int channel = 0; channel < channelCount; ++channel) {
                cout << finalMedians[channel] << " ";
            }
            cout << endl;

            // Reconstruct median image
            cv::Mat medianImage;
            cv::merge(originalChannels, medianImage);

            // Save the median image
            cout << "Processing Image " << imagePath << " with Median..." << endl;
            cv::imwrite(outputPath, medianImage);
            cout << "Median image saved as: " << outputPath << endl;

            end_time = MPI_Wtime();
            double elapsed_time = end_time - start_time;
            cout << "Median completed successfully in " << elapsed_time << " seconds. " << endl;

            // Pixel value comparison for checking
            for (int channel = 0; channel < channelCount; ++channel) {
                int originalPixelValue = originalChannels[channel].at<uchar>(pixelCount / 2);
                int medianPixelValue = finalMedians[channel];
                cout << "Channel " << channel << ": Original Pixel Value = " << originalPixelValue
                    << ", Median Pixel Value = " << medianPixelValue << endl;
            }
            cout << "Thank you for using Our Parallel Image Processing with MPI." << endl;
        }

    }
    break;
    case 5:
    {
        // to start time
        start_time = MPI_Wtime();

        // Scatter the image data
        MPI_Scatter(image.data, local_rows * cols, MPI_CHAR,
            local_image.data, local_rows * cols, MPI_CHAR,
            0, MPI_COMM_WORLD);

        // Local histogram calculation
        int localHistogram[256] = { 0 };
        for (int i = 0; i < local_image.rows; i++) {
            for (int j = 0; j < local_image.cols; j++) {
                int pixelValue = local_image.at<uchar>(i, j);
                localHistogram[pixelValue]++;
            }
        }

        // Global histogram calculation
        int globalHistogram[256] = { 0 };
        MPI_Reduce(localHistogram, globalHistogram, 256, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // Global cumulative distribution function (CDF) calculation
        int globalCDF[256] = { 0 };
        if (world_rank == 0) {
            int cumulativeSum = 0;
            for (int i = 0; i < 256; i++) {
                cumulativeSum += globalHistogram[i];
                globalCDF[i] = cumulativeSum;
            }
        }

        // Broadcast global CDF to all processes
        MPI_Bcast(globalCDF, 256, MPI_INT, 0, MPI_COMM_WORLD);

        // Histogram equalization
        for (int i = 0; i < local_image.rows; i++) {
            for (int j = 0; j < local_image.cols; j++) {
                int pixelValue = local_image.at<uchar>(i, j);
                int equalizedPixelValue = (int)((globalCDF[pixelValue] * 255.0) / (rows * cols));
                local_image.at<uchar>(i, j) = equalizedPixelValue;
            }
        }

        // make that to handle buffer size allocated for receiving 
        Mat equalizedImage;
        if (world_rank == 0) {
            equalizedImage.create(rows, cols, CV_8UC1);
        }

        // Determine total number of pixels
        int total_pixels = rows * cols;

        // Allocate memory for receiving buffer
        uchar* recv_data = nullptr;
        if (world_rank == 0) {
            recv_data = new uchar[total_pixels];
        }

        // Gather equalized image portions 
        MPI_Gather(local_image.data, local_rows * cols, MPI_CHAR,
            recv_data, local_rows * cols, MPI_CHAR,
            0, MPI_COMM_WORLD);

        // Copy received data to equalizedImage buffer
        if (world_rank == 0) {
            memcpy(equalizedImage.data, recv_data, total_pixels);
            delete[] recv_data;
        }

        // Save the final equalized image on the master
        if (world_rank == 0) {
            cout << "Processing Image " << imagePath << " with Histogram Equalization..." << endl;
            imwrite(outputPath, equalizedImage);
            cout << "Equalized Image saved as " << outputPath << endl;
            end_time = MPI_Wtime();

            double elapsed_time = end_time - start_time;
            cout << "Histogram Equalization operation completed successfully in " << elapsed_time << " seconds. " << endl;
            cout << "Thank you for using Our Parallel Image Processing with MPI." << endl;
        }
    }
       break;
    case 6:
    {
        // start time
        start_time = MPI_Wtime();
       
        // Scatter 
        MPI_Scatter(image.data + start_row * cols, local_rows * cols, MPI_CHAR,
            local_image.data, local_rows * cols, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Process local portion of the image
        cv::Mat edges;
        cv::Canny(local_image, edges, threshold1, threshold2);

        // Gather the processed images on the master
        cv::Mat global_edges;
        if (world_rank == 0) {
            global_edges.create(image.size(), CV_8U);
        }
        MPI_Gather(edges.data, edges.total(), MPI_CHAR, global_edges.data, edges.total(), MPI_CHAR, 0, MPI_COMM_WORLD);

        // Save or process the final edge-detected image on the master
        if (world_rank == 0) {
            cout << "Processing Image " << imagePath << " with Edge Detection..." << endl;
            cv::imwrite(outputPath, global_edges);
            cout << "Edge Detection Image saved as " << outputPath << endl;
            end_time = MPI_Wtime();

            double elapsed_time = end_time - start_time;
            cout << "Edge Detection operation completed successfully in " << elapsed_time << " seconds. " << endl;
            cout << "Thank you for using Our Parallel Image Processing with MPI." << endl;
            
        }
    }
        break;
    case 7:
    {
         start_time = MPI_Wtime();
        // Scatter image data
        MPI_Scatter(image.data, local_rows* cols * 3, MPI_CHAR,
            local_image.data, local_rows* cols * 3, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Perform color space conversion to grayscale on each process
        cvtColor(local_image, local_image, COLOR_BGR2GRAY);

        // Gather the converted images to rank 0
        Mat global_gray_image;
        if (world_rank == 0) {
            global_gray_image.create(image.size(), CV_8UC1);
        }
        MPI_Gather(local_image.data, local_rows * cols, MPI_CHAR, global_gray_image.data, local_rows * cols, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Perform color space conversion to HSV on each process
        cvtColor(local_image, local_image, COLOR_GRAY2BGR);
        cvtColor(local_image, local_image, COLOR_BGR2HSV);

        // Gather the converted images to rank 0
        Mat global_hsv_image;
        if (world_rank == 0) {
            global_hsv_image.create(image.size(), CV_8UC3);
        }
        MPI_Gather(local_image.data, local_rows * cols * 3, MPI_CHAR, global_hsv_image.data, local_rows * cols * 3, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Save or process the final converted images on the master
        if (world_rank == 0) {
            cout << "Processing Image " << imagePath << " with Color Space Conversion..." << endl;
            cv::imwrite("converted_gray_image.jpg", global_gray_image);
            cv::imwrite("converted_hsv_image.jpg", global_hsv_image);
          
            end_time = MPI_Wtime();
            double elapsed_time = end_time - start_time;
            cout << "Color Space Conversion operation completed successfully in " << elapsed_time << " seconds. " << endl;
            cout << "Thank you for using Our Parallel Image Processing with MPI." << endl;
        }

       
    }
        break;
    default:
        if (world_rank == 0) {
            cout << "Invalid choice. Please choose a number between 1 and 7." << endl;
        }
    }

    MPI_Finalize();

    return 0;
}
