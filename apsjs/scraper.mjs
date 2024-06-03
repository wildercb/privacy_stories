import fs from 'node:fs/promises';
import gplay from 'google-play-scraper';
import striptags from 'striptags';

// Function to create a directory if not exists
async function createDirectory(directoryPath) {
  try {
    await fs.mkdir(directoryPath, { recursive: true });
  } catch (err) {
    if (err.code !== 'EEXIST') {
      throw err;
    }
  }
}

// Function to remove HTML tags
function removeHtmlTags(text) {
  return striptags(text);
}

const countryCode = 'US';
const country = 'United States';

// Create the output directory
const outputDirectory = `output/${country}`;
createDirectory(outputDirectory);

// Create the privacy_policies directory
const privacyPoliciesDirectory = `${outputDirectory}/privacy_policies`;
createDirectory(privacyPoliciesDirectory);

// Fetch and process the top 1000 apps
const fetchAndProcessApps = async (collection) => {
  try {
    const results = await gplay.list({
      category: gplay.category.APPLICATION,
      collection: collection,
      num: 1000,
      country: countryCode,
    });

    for (let index = 0; index < results.length; index++) {
      const { appId, title } = results[index];

      try {
        const dataSafetyInfo = await gplay.datasafety({ appId });
        const privacyPolicyUrl = dataSafetyInfo.privacyPolicyUrl;

        if (privacyPolicyUrl) {
          const fileName = `${appId}_ppolicy.txt`;
          const filePath = `${privacyPoliciesDirectory}/${fileName}`;
          await fs.writeFile(filePath, privacyPolicyUrl, { encoding: 'utf-8' });
          console.log(`Privacy policy URL for ${title} (${appId}) saved to ${filePath}`);
        } else {
          console.log(`Privacy policy URL not available for ${title} (${appId})`);
        }
      } catch (error) {
        console.error(`Error fetching data safety information for ${title} (${appId}): ${error.message}`);
      }
    }
  } catch (error) {
    console.error(`Error fetching app list: ${error.message}`);
  }
};

// Fetch and process the top free apps
fetchAndProcessApps(gplay.collection.TOP_FREE);

// Fetch and process the top paid apps
fetchAndProcessApps(gplay.collection.TOP_PAID);