package NILE;

import java.io.*;
import java.util.List;

import edu.harvard.hsph.biostats.nile.InconsistentPhraseDefinitionException;
import edu.harvard.hsph.biostats.nile.NaturalLanguageProcessor;
import edu.harvard.hsph.biostats.nile.SemanticObject;
import edu.harvard.hsph.biostats.nile.SemanticRole;
import edu.harvard.hsph.biostats.nile.Sentence;

public class Example {

	public static void processEHRNotes(String inputPath, String outputPath, String textColumnName) {
		try (BufferedReader reader = new BufferedReader(new FileReader(inputPath));
			 BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
			
			// Write header
			writer.write("Source_Text,Extracted_Text,Codes,SemanticRole,Certainty,FamilyHistory\n");
			
			// Get header row and find text column index
			String[] headers = reader.readLine().split(",");
			int textColumnIndex = -1;
			for (int i = 0; i < headers.length; i++) {
				if (headers[i].equals(textColumnName)) {
					textColumnIndex = i;
					break;
				}
			}
			
			if (textColumnIndex == -1) {
				throw new RuntimeException("Text column '" + textColumnName + "' not found in CSV");
			}

			// Prepare the NLP environment
			NaturalLanguageProcessor nlp = null;
			try {
				nlp = new NaturalLanguageProcessor();
			} catch (InconsistentPhraseDefinitionException e) {
				System.err.println(e.getMessage());
				System.exit(1);
			}
			
			// Process each row
			String line;
			while ((line = reader.readLine()) != null) {
				String[] row = line.split(",");
				if (row.length > textColumnIndex) {
					String ehrText = row[textColumnIndex];
					
					// Process the EHR text
					for (Sentence s : nlp.digTextLine(ehrText)) {
						for (SemanticObject obj : s.getSemanticObjs()) {
							// Get text and clean it
							String objText = obj.getText().replace(",", ";"); // Replace commas
							
							// Get codes
							List<String> codes = obj.getCode();
							String codesStr = String.join(";", codes);
							
							// Write the row
							writer.write(String.format("\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n",
								ehrText.substring(0, Math.min(50, ehrText.length())).replace("\"", "\"\""),
								objText.replace("\"", "\"\""),
								codesStr.replace("\"", "\"\""),
								obj.getSemanticRole(),
								obj.getCertainty(),
								obj.isFamilyHistory()
							));
						}
					}
					
					// Add blank row between different notes
					writer.write("\n");
				}
			}
			
			System.out.println("CSV file has been created successfully at: " + outputPath);
			
		} catch (IOException e) {
			System.out.println("Error processing CSV: " + e.getMessage());
			e.printStackTrace();
		}
	}

	public static void printUsage() {
		System.out.println("Usage: java -cp \".:NILE.jar:opencsv-5.7.1.jar\" Example [options]");
		System.out.println("Options:");
		System.out.println("  -i, --input     Input CSV file path (required)");
		System.out.println("  -o, --output    Output CSV file path (required)");
		System.out.println("  -c, --column    Text column name (default: 'text')");
		System.out.println("\nExample:");
		System.out.println("  java -cp \".:NILE.jar:opencsv-5.7.1.jar\" Example -i data/input.csv -o data/output.csv -c note_text");
	}

	public static void main(String[] args) {
		String inputPath = null;
		String outputPath = null;
		String textColumnName = "text";  // default value

		// Parse command line arguments
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
				case "-i":
				case "--input":
					if (i + 1 < args.length) {
						inputPath = args[++i];
					}
					break;
				case "-o":
				case "--output":
					if (i + 1 < args.length) {
						outputPath = args[++i];
					}
					break;
				case "-c":
				case "--column":
					if (i + 1 < args.length) {
						textColumnName = args[++i];
					}
					break;
				case "-h":
				case "--help":
					printUsage();
					return;
				default:
					System.out.println("Unknown option: " + args[i]);
					printUsage();
					return;
			}
		}

		// Validate required parameters
		if (inputPath == null || outputPath == null) {
			System.out.println("Error: Input and output paths are required");
			printUsage();
			return;
		}

		// Validate input file exists
		File inputFile = new File(inputPath);
		if (!inputFile.exists()) {
			System.out.println("Error: Input file does not exist: " + inputPath);
			return;
		}

		System.out.println("Processing with parameters:");
		System.out.println("Input file: " + inputPath);
		System.out.println("Output file: " + outputPath);
		System.out.println("Text column: " + textColumnName);
		
		processEHRNotes(inputPath, outputPath, textColumnName);
	}
	
	static private void print(SemanticObject obj){
		System.out.println(obj.getText());
		for(SemanticObject mod: obj.getModifiers()){
			print(mod);
		}
	}

}
