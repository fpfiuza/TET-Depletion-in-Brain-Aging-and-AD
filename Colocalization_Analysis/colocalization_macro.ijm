// --- Block 0: Initial Preparation ---
//-----------------------------------------
print("\\Clear");
showMessage("Macro Start", "The macro will now begin. Follow the instructions at each step.");

// 1. Open images with example names
path_dapi = File.openDialog("Step 1: Select your DAPI image (e.g., b0c0x0)");
open(path_dapi); dapi_title = getTitle();
path_tet = File.openDialog("Step 1: Select your TET image (e.g., b0c1x0)");
open(path_tet); tet_title = getTitle();
path_neun = File.openDialog("Step 1: Select your NeuN image (e.g., b0c3x0)");
open(path_neun); neun_title = getTitle();

// 2. 8-bit and 3. Remove scale
run("8-bit"); run("Set Scale...", "distance=0 known=0 pixel=1 unit=unit");
selectWindow(tet_title);
run("8-bit"); run("Set Scale...", "distance=0 known=0 pixel=1 unit=unit");
selectWindow(dapi_title);
run("8-bit"); run("Set Scale...", "distance=0 known=0 pixel=1 unit=unit");

// 4. Subtract Background
selectWindow(tet_title);
radius = getNumber("Step 4: Enter the 'Rolling ball radius' value:", 50);
run("Subtract Background...", "rolling=" + radius);

// 5. Merge Channels
run("Merge Channels...", "c1=["+neun_title+"] c2=["+tet_title+"] c3=["+dapi_title+"] create keep");
composite_title = "Composite";
showMessage("Channels Merged", "The images have been combined into a Composite.");


// --- Block 1: Nuclear Analysis ---
//------------------------------------
waitForUser("Next Step: Nuclear Analysis", "Now we will analyze colocalization in the nuclei (DAPI).\nClick OK to continue.");
roiManager("reset");

// 6. Open DAPI ROIs
path_roi_dapi = File.openDialog("Step 6: Select the DAPI ROIs file");
roiManager("Open", path_roi_dapi);

// 7. PAUSE FOR MANUAL COMBINATION
waitForUser("Manual Action Required: Combine DAPI ROIs",
    "The DAPI ROI list has been loaded.\n\n" +
    "Do the following in the ROI Manager:\n" +
    "  1. Select ALL ROIs in the list.\n" +
    "  2. Click on 'More >>' and then 'Combine'.\n" +
    "  3. Rename the newly created ROI to 'all_dapi'.\n" +
    "  4. Create an ROI for the entire image (Edit>Selection>Select All) and add to Manager as 'whole_image'.\n\n" +
    "Click 'OK' on this box WHEN YOU ARE DONE for the macro to continue.");

// 9. Apply JaCoP
waitForUser("Action Required: Run JaCoP (Nuclear)", "Configure JaCoP (A=TET, B=DAPI), use the 'all_dapi' ROI, and save the results.");

// 11 and 12. Simplified Cleanup
if(isOpen("Results")) { selectWindow("Results"); run("Close"); }
if(isOpen("Log")) { selectWindow("Log"); run("Close"); }
roiManager("reset");
print("Nuclear Analysis cleanup completed.");


// --- Block 2: Whole Cell Analysis ---
//-----------------------------------------
waitForUser("Next Step: Whole Cell Analysis", "Now we will analyze colocalization in the whole cell (NeuN).\nClick OK to continue.");

// 13. Insert NeuN ROIs
path_roi_neun = File.openDialog("Step 13: Select the NeuN ROIs file");
roiManager("Open", path_roi_neun);

// 14. PAUSE FOR MANUAL COMBINATION
waitForUser("Manual Action Required: Combine NeuN ROIs",
    "The NeuN ROI list has been loaded.\n\n" +
    "Do the following in the ROI Manager:\n" +
    "  1. Select ALL ROIs in the list.\n" +
    "  2. Click on 'More >>' and then 'Combine'.\n" +
    "  3. Rename the newly created ROI to 'all_neun'.\n" +
    "  4. Create an ROI for the entire image (Edit>Selection>Select All) and add to Manager as 'whole_image'.\n\n" +
    "Click 'OK' on this box WHEN YOU ARE DONE.");

// 16. Apply JaCoP
waitForUser("Action Required: Run JaCoP (Whole Cell)", "Configure JaCoP (A=TET, B=NeuN), use the 'all_neun' ROI, and save the results.");

// 18 and 19. Simplified Cleanup
if(isOpen("Results")) { selectWindow("Results"); run("Close"); }
if(isOpen("Log")) { selectWindow("Log"); run("Close"); }
roiManager("reset");
print("Whole Cell Analysis cleanup completed.");


// --- Block 3: Cytoplasmic Analysis ---
//------------------------------------------
waitForUser("Next Step: Cytoplasm Analysis", "Let's create the cytoplasm ROIs.");
roiManager("reset");

// 20. Load ROIs for XOR
showMessage("Step 20: Load ROIs for XOR", "First, load the DAPI ROIs, and then the NeuN ROIs, in the same order.");
path_dapi_xor = File.openDialog("Select the DAPI ROIs file (first)");
roiManager("Open", path_dapi_xor);
path_neun_xor = File.openDialog("Now, select the NeuN ROIs file (second)");
roiManager("Open", path_neun_xor);

// XOR Macro
totalRois = roiManager("count");
if (totalRois % 2 != 0) { exit("ERROR: The total number of ROIs is not even for the XOR operation."); }
n_pares = totalRois / 2;
print("Starting XOR operation with " + n_pares + " pairs.");
for (i = 0; i < n_pares; i++) {
    index_dapi = i; index_neun = i + n_pares;
    roiManager("Select", newArray(index_dapi, index_neun));
    roiManager("XOR");
    roiManager("Add");
    roiManager("Rename", "Cytoplasm_" + (i + 1));
}
roiManager("Deselect");
print(n_pares + " cytoplasm ROIs have been created.");

// 21. PAUSE FOR MANUAL COMBINATION of Cytoplasm
waitForUser("Manual Action Required: Combine Cytoplasm ROIs",
    "The 'Cytoplasm' ROIs have been created.\n\n" +
    "Do the following in the ROI Manager:\n" +
    "  1. Select ALL 'Cytoplasm_X' ROIs.\n" +
    "  2. Click on 'More >>' and then 'Combine'.\n" +
    "  3. Rename the newly created ROI to 'all_cyto'.\n\n" +
    "Click 'OK' on this box WHEN YOU ARE DONE.");

// 22. Apply JaCoP to Cytoplasm
waitForUser("Action Required: Run JaCoP (Cytoplasm)", "Configure JaCoP (A=TET, B=NeuN), use the 'all_cyto' ROI, and save the results.");


// --- Block 4: Absolute Final Cleanup ---
//-----------------------------------------
waitForUser("Final Cleanup", "Analysis is complete. Clicking OK will close ALL windows.\n\nMake sure you have saved all 3 result files.");

print("--- Starting Absolute Final Cleanup Routine ---");

// Clear ROI Manager first and Results.
if(isOpen("Results")) { selectWindow("Results"); run("Close"); }
if(isOpen("Log")) { selectWindow("Log"); run("Close"); }

roiManager("reset");
print("ROI Manager cleared.");

// Now, close ALL open windows (images, plots, etc).
run("Close All");
print("All windows have been closed.");

showMessage("MACRO END", "The entire process has been completed successfully!");