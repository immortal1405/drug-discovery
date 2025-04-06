# Molecular AI Frontend

This is the frontend application for the Molecular AI drug discovery platform. It integrates with our molecular generation backend to enable users to generate and visualize drug-like molecules with optimized properties.

## Features

- User authentication and project management
- Molecule generation with configurable parameters
- Property optimization for QED, LogP, binding affinity, etc.
- Advanced visualization of molecular properties
- Results storage and analysis
- Interactive molecule viewer

## Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.8+ with pip (for the backend)
- PostgreSQL (optional, for persistent storage)

## Project Structure

- `/src`: Source code for the Next.js application
- `/public`: Static assets and generated results
- `/prisma`: Database schema and migrations
- `/molecular-ai`: Backend Python code for molecule generation (in a separate directory)

## Setup and Installation

### 1. Frontend Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/molecular-frontend.git
   cd molecular-frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Create a `.env` file with the following content:
   ```
   DATABASE_URL="postgresql://username:password@localhost:5432/mydb?schema=public"
   NEXTAUTH_SECRET="your-secret-key"
   NEXTAUTH_URL="http://localhost:3000"
   ```

4. (Optional) Initialize the database:
   ```
   npx prisma migrate dev
   ```

### 2. Backend Setup

1. Make sure the `molecular-ai` directory is in the same parent directory as the frontend:
   ```
   cd ..
   git clone https://github.com/your-username/molecular-ai.git
   cd molecular-ai
   ```

2. Set up a Python virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the frontend development server:
   ```
   cd molecular-frontend
   npm run dev
   ```

2. Access the application at [http://localhost:3000](http://localhost:3000)

## Using the Molecule Generation Feature

1. Log in to the application
2. Navigate to the "Generate" page
3. Configure your molecule generation parameters:
   - Number of molecules
   - Optimization targets (QED, LogP, etc.)
   - Generation method
4. Click "Generate Molecules" to start the process
5. View your results in the "Results" page
6. Analyze molecular properties and visualizations

## Integration with Backend

The frontend communicates with the molecular-ai backend through a Node.js API endpoint that:

1. Takes user input from the generation form
2. Creates a unique output directory
3. Runs the Python script with appropriate parameters
4. Processes the results and visualizations
5. Returns structured data to the frontend

## Troubleshooting

- **File permissions**: Ensure the frontend has permission to execute Python scripts and write to the output directory
- **Path issues**: Make sure the path to the molecular-ai directory is correct in the API endpoint
- **Python dependencies**: Verify all required Python packages are installed
- **Database connection**: Check your PostgreSQL connection string if using the database

## License

[MIT License](LICENSE)

## Contact

Your Name - your.email@example.com 