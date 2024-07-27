function toggleSubOptions(process) {
  const subOptions = {
    "missing-values": [
      "Fill with Mean",
      "Fill with Median",
      "Fill with Mode",
      "Drop Missing Values",
    ],
    scaling: ["Standard Scaling", "Min-Max Scaling", "Robust Scaling"],
    encoding: ["One-Hot Encoding", "Label Encoding", "Binary Encoding"],
  };

  const subOptionsContainer = document.getElementById(process);
  subOptionsContainer.innerHTML = ""; // Clear previous options

  if (
    subOptionsContainer.style.display === "none" ||
    subOptionsContainer.style.display === ""
  ) {
    subOptions[process].forEach((option) => {
      const listItem = document.createElement("li");
      const button = document.createElement("button");
      button.innerText = option;
      button.onclick = () => alert(`Selected: ${option}`);
      listItem.appendChild(button);
      subOptionsContainer.appendChild(listItem);
    });
    subOptionsContainer.style.display = "block";
  } else {
    subOptionsContainer.style.display = "none";
  }
}

function populateColumnDropdown(headers) {
  const columnDropdown = document.getElementById("column-dropdown");
  columnDropdown.innerHTML = "";

  headers.forEach((header) => {
    const option = document.createElement("option");
    option.value = header;
    option.innerText = header;
    columnDropdown.appendChild(option);
  });
}

function filterColumn() {
  const selectedColumn = document.getElementById("column-dropdown").value;
  alert(`Selected column: ${selectedColumn}`);
}

// Sample function to populate table with CSV data
// This function should be replaced with actual data fetching and processing logic
function populateTable(data) {
  const table = document.getElementById("data-table");
  table.innerHTML = "";

  // Create table headers
  const headers = data[0];
  let headerRow = "<tr>";
  headers.forEach((header) => {
    headerRow += `<th>${header}</th>`;
  });
  headerRow += "</tr>";
  table.innerHTML += headerRow;

  // Populate column dropdown
  populateColumnDropdown(headers);

  // Create table rows
  data.slice(1).forEach((row) => {
    let rowHTML = "<tr>";
    row.forEach((cell) => {
      rowHTML += `<td>${cell}</td>`;
    });
    rowHTML += "</tr>";
    table.innerHTML += rowHTML;
  });
}

// Sample data for testing
const sampleData = [
  ["Name", "Age", "City"],
  ["Alice", "25", "New York"],
  ["Bob", "30", "Los Angeles"],
  ["Charlie", "35", "Chicago"],
];

populateTable(sampleData);
