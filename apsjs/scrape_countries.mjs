import fs from 'fs';
import gplay from 'google-play-scraper';
import appStore from 'app-store-scraper';
import striptags from 'striptags';

// Set the store to scrape (either 'google' or 'apple')
const store = 'google';


// Function to log to a file
function logToFile(data, filePath) {
  fs.appendFileSync(filePath, `${data}\n`, { encoding: 'utf-8', flag: 'w' });
}

// Function to log to a JSON file
function logToJsonFile(data, filePath) {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), { encoding: 'utf-8', flag: 'w' });
}
// Function to extract and log relevant information
async function logAppDetails(result, index, filePath, isFree, jsonDetails, country, countryCode) {
  const { title, appId, description, summary, privacyPolicy, url, icon } = result;

  if (store === 'apple') {
    title = result.trackName || 'Not available';
    appId = result.bundleId || 'Not available';
    description = removeHtmlTags(result.description || summary || 'Not available');
    privacyPolicyUrl = result.privacyPolicyUrl || 'Not available';
    url = result.trackViewUrl || 'Not available';
    icon = result.artworkUrl512 || 'Not available';
  }

  const appDetails = {
    index,
    title: title || 'Not available',
    appId: appId || 'Not available',
    description: removeHtmlTags(description || summary || 'Not available'),
    privacyPolicy: privacyPolicy || 'Not available',
    url: url || 'Not available',
    isFree,
    countryCode,
  };

  // Fetch data safety information (includes privacy policy in Nov 2023) for Google Play Store
  if (store === 'google') {
    try {
      const dataSafetyInfo = await gplay.datasafety({ appId });
      appDetails.dataSafetyInfo = dataSafetyInfo;
    } catch (error) {
      console.error(`Error fetching data safety information for ${appId}: ${error.message}`);
    }
  }

  logToFile(`Index: ${appDetails.index}`, filePath);
  logToFile(`Title: ${appDetails.title}`, filePath);
  logToFile(`App ID: ${appDetails.appId}`, filePath);
  logToFile(`Description: ${appDetails.description}`, filePath);
  logToFile(`URL: ${appDetails.url}`, filePath);
  logToFile(`Is Free: ${appDetails.isFree}`, filePath);
  logToFile(`Country Code: ${appDetails.countryCode}`, filePath);
  

  // Include the privacy policy URL from data safety information in the JSON details
  if (appDetails.dataSafetyInfo && appDetails.dataSafetyInfo.privacyPolicyUrl) {
    const privacyPolicyUrlValue = appDetails.dataSafetyInfo.privacyPolicyUrl;
    logToFile(`Privacy Policy URL: ${privacyPolicyUrlValue}`, filePath);
    appDetails.privacyPolicy = privacyPolicyUrlValue; // Overwrite the original privacyPolicy with the URL
  }

  jsonDetails.push(appDetails);

  // Create a folder for each app in the respective "free" or "paid" directory
  const appFolder = `output/${country}/${isFree ? 'free' : 'paid'}/${appDetails.appId}`;
  createDirectory(appFolder);

  // Log the description to a file in the app folder
  const descriptionFilePath = `${appFolder}/description.txt`;
  logToFile(`Description: ${appDetails.description}`, descriptionFilePath);

  // Log data safety information to a file (for Google Play Store)
  if (store === 'google' && appDetails.dataSafetyInfo) {
    const dataSafetyFilePath = `${appFolder}/data_safety.txt`;
    logToFile(`Data Safety Information for ${appDetails.title} (${appDetails.appId}):`, dataSafetyFilePath);
    logToFile(JSON.stringify(appDetails.dataSafetyInfo, null, 2), dataSafetyFilePath);
  }
}


// Function to create a directory if not exists
function createDirectory(directoryPath) {
  if (!fs.existsSync(directoryPath)) {
    fs.mkdirSync(directoryPath, { recursive: true });
  }
}

// Function to remove HTML tags
function removeHtmlTags(text) {
  return striptags(text);
}

// Define EU countries and their codes
const euCountries = {
  'AT': 'Austria',
  'BE': 'Belgium',
  'BG': 'Bulgaria',
  'CY': 'Cyprus',
  'CZ': 'Czech Republic',
  'DE': 'Germany',
  'DK': 'Denmark',
  'EE': 'Estonia',
  'GR': 'Greece',
  'ES': 'Spain',
  'FI': 'Finland',
  'FR': 'France',
  'HR': 'Croatia',
  'HU': 'Hungary',
  'IE': 'Ireland',
  'IT': 'Italy',
  'LT': 'Lithuania',
  'LU': 'Luxembourg',
  'LV': 'Latvia',
  'MT': 'Malta',
  'NL': 'Netherlands',
  'PL': 'Poland',
  'PT': 'Portugal',
  'RO': 'Romania',
  'SE': 'Sweden',
  'SI': 'Slovenia',
  'SK': 'Slovakia',
};

// Loop over each EU country
Object.entries(euCountries).forEach(async ([countryCode, country]) => {
  const countryDirectory = `output/${country}`;
  createDirectory(countryDirectory);

  // Log for top_free collection
  const topFreeFilePath = `${countryDirectory}/${country}_top_100_top_free.json`;
  const jsonFreeDetails = [];

  try {
    const results = await gplay.list({
      category: gplay.category.APPLICATION,
      collection: gplay.collection.TOP_FREE,
      num: 100,
      country: countryCode,
    });

    logToFile(`Top Free Apps in ${country}`, topFreeFilePath);
    for (let index = 0; index < results.length; index++) {
      await logAppDetails(results[index], index + 1, topFreeFilePath, true, jsonFreeDetails, country, countryCode);
    }

    // Log to JSON file after processing all items
    logToJsonFile(jsonFreeDetails, topFreeFilePath);
  } catch (error) {
    logToFile(JSON.stringify(error), topFreeFilePath);
  }

  // Log for top_paid collection
  const topPaidFilePath = `${countryDirectory}/${country}_top_100_top_paid.json`;
  const jsonPaidDetails = [];

  try {
    const results = await gplay.list({
      category: gplay.category.APPLICATION,
      collection: gplay.collection.TOP_PAID,
      num: 100,
      country: countryCode,
    });

    logToFile(`Top Paid Apps in ${country}`, topPaidFilePath);
    for (let index = 0; index < results.length; index++) {
      await logAppDetails(results[index], index + 1, topPaidFilePath, false, jsonPaidDetails, country, countryCode);
    }

    // Log to JSON file after processing all items
    logToJsonFile(jsonPaidDetails, topPaidFilePath);
  } catch (error) {
    logToFile(JSON.stringify(error), topPaidFilePath);
  }
});
