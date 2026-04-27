// -----------------------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------------------
// ---------------------------------- Variational Fracture Code ----------------------------------
// Algorithm will be written here shortly.
// -----------------------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------------------

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <mfem.hpp>
#include <mfemplus.hpp>
#include <nlohmann/json.hpp>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace mfem;
using namespace std;
using json = nlohmann::json;

struct matProps
{
    double lameConstant, shearModulus;
    double Ey, nu;
    double Gc;
    double eps;
    double Kepsilon;
};

struct simProps
{
    int N;                      // Number of steps
    double errorTol;            // error tolerance for convergence of damage field.
    int maxIterations;          // maximum number of iterations, i.e. number of boundary displacements specified
    vector<float> boundaryDisp; // displacement boundary conditions.
    double eta, Deltat;         // viscosity and time increment.
};

std::tuple<int, int, int> GetNodeInfo(const mfem::GridFunction *nodes_ptr);
bool readInputParameters(int argc, char *argv[], json &inputParameters);
std::error_code logInFractureSim(json &inputParameters, matProps &fractureMatProps, simProps &fractureSimProps, mfem::ParMesh *mesh);

class BoundaryConditions
{

private:
    json inputFile;
    mfem::ParMesh *pmesh;
    mfem::ParFiniteElementSpace *fespacebc;
    mfem::Array<int> ess_bdr_x, ess_bdr_y, ess_bdr_z;
    mfem::Array<int> ess_tdof_list_x, ess_tdof_list_y, ess_tdof_list_z;
    const mfem::GridFunction *node_coords;
    vector<float> boundaryDisp;

protected:
    double hat_u_x_func(const mfem::Vector &pt, int &step);
    double hat_u_y_func(const mfem::Vector &pt, int &step);
    double hat_u_z_func(const mfem::Vector &pt, int &step);

public:
    BoundaryConditions(const json &inputParameters, mfem::ParMesh *mesh_in, mfem::ParFiniteElementSpace *fespace_bc, vector<float> &boundary_displacements) : inputFile(inputParameters), pmesh(mesh_in), fespacebc(fespace_bc), boundaryDisp(boundary_displacements)
    {
        node_coords = pmesh->GetNodes();
    };
    int determineDirichletDof(mfem::Array<int> &ess_tdof_listx, mfem::Array<int> &ess_tdof_listy, mfem::Array<int> &ess_tdof_listz, mfem::ParGridFunction &hat_u_x_nodes, mfem::ParGridFunction &hat_u_y_nodes, mfem::ParGridFunction &hat_u_z_nodes);
    int projectDirichletVals(mfem::Array<int> &ess_tdof_listx, mfem::Array<int> &ess_tdof_listy, mfem::Array<int> &ess_tdof_listz, mfem::Vector &hat_u_x, mfem::Vector &hat_u_y, mfem::Vector &hat_u_z, mfem::GridFunction &disp_gf, int step);
};

class PhaseFieldSolver
{
protected:
    json inputFile;
    Array<int> ess_tdof_list;
    const mfem::GridFunction *node_coords;
    ParGridFunction *u, *d, *h; // u is H1(B, R3), d is H1(B, R), h is L2(B,R).
    ParMesh *pmesh;
    ParFiniteElementSpace *dispfespace, *damagefespace;
    ParBilinearForm *kdisp, *kdmg1, *kdmg2;
    ParLinearForm *fdisp, *fdmg;
    HypreParMatrix *KdispMat, *KdmgMat, *Kdmg1Mat, *Kdmg2Mat;
    HypreParVector *FdispVec, *FdmgVec;
    HypreParVector *u_new, *d_vec2, *d_max, *d_new;
    HypreBoomerAMG *KdispPrec, *KdmgPrec;
    HyprePCG *KdispCG, *KdmgCG;
    double local_damage_error, global_damage_error;

public:
    PhaseFieldSolver(ParGridFunction &disp_gf, ParGridFunction &damage_gf) : u(&disp_gf), d(&damage_gf) {};
    void ReadFESpacesMeshEssDofs(ParFiniteElementSpace *disp_fes, ParFiniteElementSpace *damage_fes, ParMesh *pmesh_in, Array<int> &ess_dofs)
    {
        dispfespace = disp_fes;
        damagefespace = damage_fes;
        pmesh = pmesh_in;
        ess_tdof_list = ess_dofs;
        node_coords = pmesh->GetNodes();
    };
    // Read bilinear and linear forms.
    void ReadBilinearLinearForms(ParBilinearForm *k_disp, ParBilinearForm *k_damage1, ParBilinearForm *k_damage2, ParLinearForm *f_disp, ParLinearForm *f_damage)
    {
        kdisp = k_disp;
        kdmg1 = k_damage1;
        kdmg2 = k_damage2;
        fdisp = f_disp;
        fdmg = f_damage;
    };

    void InitializeMatricesVectors(); // Initializes HypreParMatrices for bilinar forms. Initializes HypreParVectors for linear forms.
    void InitializeSolvers();
    void AssembleKdispMatFdispVec();
    void ComputeDisp();
    void AssembleKdmg1Mat();
    void ComputeHistoryVariable();
    void AssembleKdmgMatFdmgVec();
    void ComputeDamage();
    double ComputeDamageError();
    void SuppressBoundaryDamage();
    ~PhaseFieldSolver()
    {
        delete KdispMat;
        delete Kdmg1Mat;
        delete Kdmg2Mat;
        delete FdispVec;
        delete FdmgVec;
        delete u_new;
        delete d_max;
        delete d_vec2;
        delete d_new;
        delete KdispPrec;
        delete KdmgPrec;
        delete KdispCG;
        delete KdmgCG;
    }
};

int main(int argc, char *argv[])
{
    // Uncomment the following lines to debug several mpi processes at once.
    // {
    // 	volatile int i = 0;
    // 	char hostname[256];
    // 	gethostname(hostname, sizeof(hostname));
    // 	printf("PID %d on %s ready for attach\n", getpid(), hostname);
    // 	fflush(stdout);
    // 	while (0 == i)
    // 		sleep(5);
    // }

    //------------------------------------------------------------------------------------------------------------------
    // Initialize MPI, HYPRE, json, simProps and matProps
    //------------------------------------------------------------------------------------------------------------------

    Mpi::Init(argc, argv);
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();
    Device device("cpu");

    StopWatch total_time;
    total_time.Start();

    json inputParameters;
    simProps fractureSimProps;
    matProps fractureMatProps;
    readInputParameters(argc, argv, inputParameters); // read on every MPI rank.

    std::string mesh_file_str = inputParameters["Simulation Parameters"]["Mesh Parameters"]["meshFileName"];
    const char *mesh_file = mesh_file_str.c_str(); // open mesh file on every MPI rank. Create parallel mesh and delete mesh.
    int order = inputParameters["Simulation Parameters"]["Mesh Parameters"]["order"];
    int ref_levels = inputParameters["Simulation Parameters"]["Mesh Parameters"]["ref_levels"];
    // Add other relevant parameters.

    //------------------------------------------------------------------------------------------------------------------
    // Read mesh and log in simulation info.
    //------------------------------------------------------------------------------------------------------------------

    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();      // dim should equal 3.
    mesh->SetCurvature(order, false); // Enable high-order geometry
    for (int l = 0; l < ref_levels; l++)
        mesh->UniformRefinement();

    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    logInFractureSim(inputParameters, fractureMatProps, fractureSimProps, pmesh);

    //------------------------------------------------------------------------------------------------------------------
    // Define finite element spaces.
    //------------------------------------------------------------------------------------------------------------------
    // We are using elements close to Taylor-Hood elements.
    // P_{k} vector Lagrange elements for displacement and P_{k} scalar lagrage elements for damage;

    FiniteElementCollection *Lagrangefec(new H1_FECollection(order, dim, BasisType::GaussLobatto));
    ParFiniteElementSpace *dispfespace, *damagefespace;

    dispfespace = new ParFiniteElementSpace(pmesh, Lagrangefec, dim); // Define a vector H1 finite element space for nodal displacements.
    damagefespace = new ParFiniteElementSpace(pmesh, Lagrangefec, 1); // Define a scalar H1 finite element space for nodal pressures.
    pmesh->SetNodalFESpace(dispfespace);
    pmesh->EnsureNodes();

    // Define L2 finite element space for history variable.
    FiniteElementCollection *L2fec(new L2_FECollection(0, dim));
    ParFiniteElementSpace *historyfespace = new ParFiniteElementSpace(pmesh, L2fec, 1);

    if (myid == 0)
        cout << "Initialized finite element spaces" << endl;
    //------------------------------------------------------------------------------------------------------------------
    // Define and initialize all grid functions
    //------------------------------------------------------------------------------------------------------------------

    ParGridFunction u(dispfespace), d(damagefespace);
    u = 0.0;
    d = 0.0;

    ParGridFunction h(historyfespace); // h is the grid function for the history variable for each element, not node.
    h = 0.0;

    ParGridFunction hat_u_x_nodes(damagefespace), hat_u_y_nodes(damagefespace), hat_u_z_nodes(damagefespace);
    hat_u_x_nodes = 0.0;
    hat_u_y_nodes = 0.0;
    hat_u_z_nodes = 0.0;

    if (myid == 0)
        cout << "Initialized grid functions" << endl;

    //------------------------------------------------------------------------------------------------------------------

    pmesh->EnsureNodes();
    GridFunction *nodes = pmesh->GetNodes();
    Array<int> glob_elem_ids;
    pmesh->GetGlobalElementIndices(glob_elem_ids);
    const int vdim = nodes->VectorDim(); // Should be 3 for 2D
    const int num_nodes = nodes->Size() / vdim;
    const int num_els = dispfespace->GetNE();

    //------------------------------------------------------------------------------------------------------------------
    // Read boundary attributes and determine essential (Dirichlet) dofs.
    //------------------------------------------------------------------------------------------------------------------
    BoundaryConditions BCs(inputParameters, pmesh, dispfespace, fractureSimProps.boundaryDisp);

    Array<int> ess_tdof_listbcx, ess_tdof_listbcy, ess_tdof_listbcz, ess_tdof_list;
    BCs.determineDirichletDof(ess_tdof_listbcx, ess_tdof_listbcy, ess_tdof_listbcz, hat_u_x_nodes, hat_u_y_nodes, hat_u_z_nodes);
    ess_tdof_list.Append(ess_tdof_listbcx);
    ess_tdof_list.Append(ess_tdof_listbcy);
    if (dim == 3)
        ess_tdof_list.Append(ess_tdof_listbcz);
    Vector hat_u_x(ess_tdof_listbcx.Size()), hat_u_y(ess_tdof_listbcy.Size()), hat_u_z(ess_tdof_listbcz.Size());
    BCs.projectDirichletVals(ess_tdof_listbcx, ess_tdof_listbcy, ess_tdof_listbcz, hat_u_x, hat_u_y, hat_u_z, u, 0); // Can later modify the functions to remove redundant arguments.

    if (myid == 0)
        cout << "Initialized boundary conditions object" << endl;

    //------------------------------------------------------------------------------------------------------------------
    // Bilinear and Linear forms.
    //------------------------------------------------------------------------------------------------------------------

    // Stiffness matrix for displacement (K_{u}) is the elastic strain energy multiplied by degradation function (1 - d)^{2} + k_{\varepsilon}
    // Stiffness matrix and corresponding boundary dislacements load term is recomputed at every iteration and sub iteration.

    // Stiffness matrix 1 for damage (K_{d1}) is computed only once.
    // Stiffness matrix 2 for damage (K_{d2}) is computed every sub iteration.
    // Load vector for damage (F_{d}) is computed every sub iteration.

    // In total:
    // 1 bilinear form and 1 linear form for displacement.
    // 2 bilinear forms and 1 linear for for damage.
    // Set up bilinear forms with integrators and pass pointers to the solver class. In the class the bilinear forms will be assembled as required.

    //------------------------------------------------------------------------------------------------------------------
    // Bilinear forms

    // kdisp
    ConstantCoefficient Ey_coeff(fractureMatProps.Ey), nu_coeff(fractureMatProps.nu);
    // Computes degradation term at each quadrature point.
    ParBilinearForm *kdisp(new ParBilinearForm(dispfespace));
    kdisp->AddDomainIntegrator(new mfemplus::IsotropicElasticityDamageIntegrator(Ey_coeff, nu_coeff, fractureMatProps.Kepsilon, d, damagefespace));

    // kdmg1
    // Viscosity term added here, add to kdmg2 instead if \Delta t changes with each time step since kdmg1 is not recomputed.
    ConstantCoefficient g_over_eps_coeff(fractureMatProps.Gc / fractureMatProps.eps), g_times_eps_coeff(fractureMatProps.Gc * fractureMatProps.eps);
    ConstantCoefficient viscosity_coeff(fractureSimProps.eta / fractureSimProps.Deltat);
    // Assembled only once in the simulation.
    ParBilinearForm *kdmg1(new ParBilinearForm(damagefespace));
    kdmg1->AddDomainIntegrator(new MassIntegrator(g_over_eps_coeff));
    kdmg1->AddDomainIntegrator(new DiffusionIntegrator(g_times_eps_coeff));
    // kdmg1->AddDomainIntegrator(new MassIntegrator(viscosity_coeff));

    // kdmg2
    // Computes the elastic energy at each quadrature point.
    ParBilinearForm *kdmg2(new ParBilinearForm(damagefespace));
    kdmg2->AddDomainIntegrator(new mfemplus::IsotropicStrainEnergyDamageIntegrator(Ey_coeff, nu_coeff, u, dispfespace));

    //------------------------------------------------------------------------------------------------------------------
    // Linear forms

    // fdisp
    ParLinearForm *fdisp(new ParLinearForm(dispfespace));
    // If growth term is needed, include a body force.
    // fdisp->AddDomainIntegrator(new DomainLFIntegrator(expansionCoeff));

    // fdmg
    // Computes the elastic energy at each quadrature point. Viscosity term also added to the same integrator to save assembly time. Viscosity computed with damage field at each quadrature point.
    // viscosity is \frac{\eta}{\Delta t} \phi
    ParLinearForm *fdmg(new ParLinearForm(damagefespace));
    fdmg->AddDomainIntegrator(new mfemplus::FractureDamageLFIntegrator(Ey_coeff, nu_coeff, viscosity_coeff, u, d, dispfespace));
    // Viscosity turned off currently in mfemplus.

    if (myid == 0)
        cout << "Initialized bilinear and linear forms." << endl;
    // -----------------------------------------------------------------------------------------------------------------
    // Set up object to save data
    std::string resultsFolder = "../results/" + inputParameters["testName"].get<std::string>();

    adios2stream aos(resultsFolder + "/WavesMFEMoutput.bp", adios2stream::openmode::out, MPI_COMM_WORLD, "BP5");
    {
        aos.SetRefinementLevel(0);
        aos.SetParameter("FlushStepsCount", "5");
        aos.BeginStep();
        pmesh->Print(aos);
        aos.SetCycle(0);
        // Write fields
        u.Save(aos, "disp", adios2stream::data_type::point_data);
        d.Save(aos, "damage", adios2stream::data_type::point_data);
        hat_u_x_nodes.Save(aos, "hat_u_x_nodes", adios2stream::data_type::point_data);
        hat_u_y_nodes.Save(aos, "hat_u_y_nodes", adios2stream::data_type::point_data);
        hat_u_z_nodes.Save(aos, "hat_u_z_nodes", adios2stream::data_type::point_data);
        aos.EndStep();
    }

    if (myid == 0)
        cout << "Initialized output stream object." << endl;

    //------------------------------------------------------------------------------------------------------------------
    // Initialize the main phase field solver object and begin iterations.
    //------------------------------------------------------------------------------------------------------------------

    PhaseFieldSolver PhaseFieldFractureSolver(u, d);
    PhaseFieldFractureSolver.ReadFESpacesMeshEssDofs(dispfespace, damagefespace, pmesh, ess_tdof_list);
    PhaseFieldFractureSolver.ReadBilinearLinearForms(kdisp, kdmg1, kdmg2, fdisp, fdmg);
    PhaseFieldFractureSolver.InitializeMatricesVectors();
    PhaseFieldFractureSolver.InitializeSolvers();

    double damage_error = 1.0;
    int iterations = 0;

    if (myid == 0)
        cout << "Initialized fracture solver. Loop starting." << endl;

    for (int step = 1; step < fractureSimProps.N; step++)
    {
        // Kdmg1Mat is only assembled once during the entire simulation.
        if (step == 1)
            PhaseFieldFractureSolver.AssembleKdmg1Mat();

        damage_error = 1.0;
        iterations = 0;

        while (damage_error >= fractureSimProps.errorTol && iterations <= fractureSimProps.maxIterations)
        {
            if (myid == 0)
                cout << "Step " << step << " iteration " << iterations << endl;

            // Project essential bcs.
            BCs.projectDirichletVals(ess_tdof_listbcx, ess_tdof_listbcy, ess_tdof_listbcz, hat_u_x, hat_u_y, hat_u_z, u, step);

            // Assemble KdispMat and FdispVec and solve for displacement.
            PhaseFieldFractureSolver.AssembleKdispMatFdispVec();
            PhaseFieldFractureSolver.ComputeDisp();

            // Assemble KdmgMat and FdmgVec and solve for displacement.
            PhaseFieldFractureSolver.AssembleKdmgMatFdmgVec();
            PhaseFieldFractureSolver.ComputeDamage();
            PhaseFieldFractureSolver.SuppressBoundaryDamage();

            // Compute error in damage.
            damage_error = PhaseFieldFractureSolver.ComputeDamageError();
            iterations++;
        };

        if (myid == 0)
            cout << "Step " << step << " took " << iterations << " iterations" << endl;

        aos.BeginStep();
        pmesh->Print(aos);
        if (myid == 0)
        {
            aos.SetCycle(step);
        }
        u.Save(aos, "disp", adios2stream::data_type::point_data);
        d.Save(aos, "damage", adios2stream::data_type::point_data);
        aos.EndStep();

        if (iterations >= fractureSimProps.maxIterations && myid == 0)
        {
            cout << "Failed. Damage field did not converge!" << endl;
            // kill program.
            std::error_code ec;
            std::cerr << "Failed. Damage field did not converge!" << ec.message() << std::endl;
            _exit(2);
        }
        if (myid == 0)
            cout << "Completed step " << step << endl;
    }

    //------------------------------------------------------------------------------------------------------------------
    // Release memory.
    //-----------------------

    total_time.Stop();
    if (myid == 0)
        cout << "Simulation completed. Took " << total_time.RealTime() << " seconds." << endl;

    delete pmesh;
    delete Lagrangefec;
    delete dispfespace;
    delete damagefespace;
    delete kdisp;
    delete kdmg1;
    delete kdmg2;
    delete fdisp;
    delete fdmg;
    return 0;
};

//----------------------------------------------------------------------------------------------------------------------------------------------
// Auxiliary functions defined here.
//----------------------------------------------------------------------------------------------------------------------------------------------

std::tuple<int, int, int> GetNodeInfo(const mfem::GridFunction *nodes_ptr)
{
    if (!nodes_ptr)
    {
        throw std::invalid_argument("Error: nodes_ptr is null.");
    }

    int num_dofs = nodes_ptr->Size();  // total number of entries
    int vdim = nodes_ptr->VectorDim(); // dimension of coordinates
    int num_nodes = num_dofs / vdim;   // number of actual nodes in this partition.

    return std::make_tuple(num_dofs, vdim, num_nodes);
}

bool readInputParameters(int argc, char *argv[], json &inputParameters)
{

    int myid = Mpi::WorldRank();
    if (myid == 0)
        std::cout << "Inside the readInputParameters file" << std::endl;
    //  (argc > 1) ? argv[1] :
    std::string json_path = "../input/inputParameters/VariationalFracture_original.json"; // Change JSON file path accordingly.

    std::ifstream infile(json_path);
    if (!infile.is_open())
    {
        std::cerr << "Error: Could not open " << json_path << std::endl;
        return 1;
    }

    infile >> inputParameters;

    return 0;
}

std::error_code logInFractureSim(json &inputParameters, matProps &fractureMatProps, simProps &fractureSimProps, mfem::ParMesh *mesh)
{
    int myid = Mpi::WorldRank();
    int numids = Mpi::WorldSize();

    std::string resultsFolder = "../results/" + inputParameters["testName"].get<std::string>();
    // std::string resultsFolder = "/users/akulaka1/scratch/MFEMresults/" + inputParameters["testName"].get<std::string>();

    // Only create folders on master rank to avoid conflicts
    if (myid == 0)
    {
        if (fs::exists(resultsFolder))
        {
            std::error_code ec;
            fs::remove_all(resultsFolder, ec); // remove all contents recursively
            if (ec)
            {
                std::cerr << "Error removing folder: " << ec.message() << std::endl;
                return ec;
            }
        }
        // Recreate a fresh empty directory
        fs::create_directories(resultsFolder);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    fractureMatProps.lameConstant = inputParameters["Physical Parameters"]["Lame Constant"];
    fractureMatProps.shearModulus = inputParameters["Physical Parameters"]["Shear Modulus"];
    fractureMatProps.Gc = inputParameters["Physical Parameters"]["Fracture Toughness"];
    fractureMatProps.eps = inputParameters["Physical Parameters"]["epsilon"];
    fractureMatProps.Kepsilon = inputParameters["Physical Parameters"]["Kepsilon"];

    double E, NU;
    E = (fractureMatProps.shearModulus * (3 * fractureMatProps.lameConstant + 2 * fractureMatProps.shearModulus)) / (fractureMatProps.lameConstant + fractureMatProps.shearModulus);
    NU = fractureMatProps.lameConstant / (2 * (fractureMatProps.lameConstant + fractureMatProps.shearModulus));
    double h = mesh->GetElementSize(1, /*type=*/1);

    fractureMatProps.Ey = E;
    fractureMatProps.nu = NU;

    if (myid == 0)
        cout << "Physical parameters read" << endl;

    fractureSimProps.errorTol = inputParameters["Simulation Parameters"]["Error Tolerance"];
    fractureSimProps.maxIterations = inputParameters["Simulation Parameters"]["Maximum Iterations"];
    fractureSimProps.boundaryDisp = inputParameters["Simulation Parameters"]["Boundary Displacements"].get<std::vector<float>>();
    fractureSimProps.N = (fractureSimProps.boundaryDisp).size();
    fractureSimProps.eta = inputParameters["Simulation Parameters"]["Viscosity"];
    fractureSimProps.Deltat = inputParameters["Simulation Parameters"]["Time Increment"];

    if (myid == 0)
        cout
            << "Simulation parameters read" << endl;

    int total_elements(0), local_elements(mesh->GetNE());
    MPI_Allreduce(&local_elements, &total_elements, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // Save simulation details on master rank
    if (myid == 0)
    {
        ofstream u_data(resultsFolder + "/" + "SimulationDetails.txt");

        u_data << "saved output to " + resultsFolder << endl;

        u_data << "Ey\t" << E << endl;
        u_data << "nu\t" << NU << endl;
        u_data << "Number of steps \t" << fractureSimProps.N << endl;
        u_data << "\n\n";
        u_data << "Mesh Details";
        u_data << "\n\n";
        u_data << "order: " << inputParameters["Simulation Parameters"]["Mesh Parameters"]["order"] << "\n";
        u_data << "ref_levels: " << inputParameters["Simulation Parameters"]["Mesh Parameters"]["ref_levels"] << "\n";
        u_data << "Number of elements: " << total_elements << "\n";

        u_data.close();
    }

    return {};
};

//----------------------------------------------------------------------------------------------------------------------------------------------
// BoundaryConditions methods defined here.
//----------------------------------------------------------------------------------------------------------------------------------------------
int BoundaryConditions::determineDirichletDof(mfem::Array<int> &ess_tdof_listx, mfem::Array<int> &ess_tdof_listy, mfem::Array<int> &ess_tdof_listz, mfem::ParGridFunction &hat_u_x_nodes, mfem::ParGridFunction &hat_u_y_nodes, mfem::ParGridFunction &hat_u_z_nodes)
{
    // Every rank needs to have the same lists!
    int myid = Mpi::WorldRank();
    int dim = pmesh->Dimension();

    mfem::Array<int> bdr_nodes_list_x, bdr_nodes_list_y;
    mfem::Array<int> bdr_nodes_list_z;

    ess_bdr_x.SetSize(pmesh->bdr_attributes.Max());
    ess_bdr_y.SetSize(pmesh->bdr_attributes.Max());
    ess_bdr_z.SetSize(pmesh->bdr_attributes.Max());

    ess_bdr_x = 0;
    ess_bdr_y = 0;
    ess_bdr_z = 0;

    // Select Dirichlet components.
    ess_bdr_x[0] = 1;
    ess_bdr_y[0] = 1;
    ess_bdr_z[0] = 1;

    ess_bdr_x[1] = 1;

    fespacebc->GetEssentialTrueDofs(ess_bdr_x, ess_tdof_listx, 0);
    fespacebc->GetEssentialTrueDofs(ess_bdr_y, ess_tdof_listy, 1);
    fespacebc->GetEssentialTrueDofs(ess_bdr_z, ess_tdof_listz, 2);

    // fespacebc->GetEssentialTrueDofs(ess_bdr_x, bdr_nodes_list_x, 0);
    // fespacebc->GetEssentialTrueDofs(ess_bdr_y, bdr_nodes_list_y, 1);
    // fespacebc->GetEssentialTrueDofs(ess_bdr_z, bdr_nodes_list_z, 2);

    // GridFunction nodesx(fespacebc), nodesy(fespacebc);
    // GridFunction nodesz(fespacebc); // scalar valued fespace.
    // nodesx = 0.0;
    // nodesy = 0.0;
    // nodesz = 0.0; // initialize, important.
    // ConstantCoefficient one(1.0);
    // nodesx.ProjectBdrCoefficient(one, ess_bdr_x);
    // nodesy.ProjectBdrCoefficient(one, ess_bdr_y);
    // nodesz.ProjectBdrCoefficient(one, ess_bdr_z);

    // for (int i = 0; i < nodesx.Size(); i++)
    // {
    //     if (nodesx(i) != 0.0)
    //         bdr_nodes_list_x.Append(i);
    //     if (nodesy(i) != 0.0)
    //         bdr_nodes_list_y.Append(i);
    //     if (nodesz(i) != 0.0)
    //         bdr_nodes_list_z.Append(i);
    // }

    // auto [num_dofs, vdim, num_nodes] = GetNodeInfo(node_coords);

    // MFEM_VERIFY(nodes, "Mesh has no nodes. Did you call SetCurvature or SetNodalFESpace?");
    // mfem::Vector pt(dim);

    // for (auto nd : bdr_nodes_list_x)
    // {
    // pt(0) = (*node_coords)(nd);
    // pt(1) = (*node_coords)(nd + num_nodes);
    // if (dim == 3)
    //     pt(2) = (*node_coords)(nd + 2 * num_nodes);
    // if (pt(1) <= 0.0001)
    // {
    // // ess_tdof_listx.Append(nd);
    // hat_u_x_nodes(nd) = 1.0;
    //     }
    // }

    // for (auto nd : bdr_nodes_list_y)
    // {
    // pt(0) = (*node_coords)(nd);
    // pt(1) = (*node_coords)(nd + num_nodes);
    // if (dim == 3)
    //     pt(2) = (*node_coords)(nd + 2 * num_nodes);
    // if (pt(1) <= 0.0001 || pt(1) >= (0.04 - 0.0001)) // Bottom and top boundaries. Don't use manual coord specification. Automate.
    // {
    // ess_tdof_listy.Append(nd + num_nodes);
    // hat_u_y_nodes(nd) = 1.0;
    // }
    // }

    // if (dim == 3)
    // {
    //     for (auto nd : bdr_nodes_list_z)
    //     {
    // pt(0) = (*node_coords)(nd);
    // pt(1) = (*node_coords)(nd + num_nodes);
    // if (dim == 3)
    //     pt(2) = (*node_coords)(nd + 2 * num_nodes);
    // if (pt(1) <= 0.0001)
    // {
    // ess_tdof_listz.Append(nd + 2 * num_nodes);
    // hat_u_z_nodes(nd) = 1.0;
    //         }
    //     }
    // }

    return 0;
}
// Boudary conditions need to be a function of time.
double BoundaryConditions::hat_u_x_func(const mfem::Vector &pt, int &step)
{
    return 0.0; // Homogeneous Dirichlet condition.
}

double BoundaryConditions::hat_u_y_func(const mfem::Vector &pt, int &step)
{
    if (pt(1) >= (0.04 - 0.0001) /*&& std::abs(pt(1) - 0.05) <= 0.0001*/)
        return -boundaryDisp[step]; // Step dependent inhomogeneous Dirichlet condition
    else
        return 0.0;
}

double BoundaryConditions::hat_u_z_func(const mfem::Vector &pt, int &step)
{
    return 0.0; // Homogeneous Dirichlet condition
}

int BoundaryConditions::projectDirichletVals(mfem::Array<int> &ess_tdof_listx, mfem::Array<int> &ess_tdof_listy, mfem::Array<int> &ess_tdof_listz, mfem::Vector &hat_u_x, mfem::Vector &hat_u_y, mfem::Vector &hat_u_z, mfem::GridFunction &disp_gf, int step)
{
    int dim = pmesh->Dimension();
    // auto [num_dofs, vdim, num_nodes] = GetNodeInfo(node_coords);
    disp_gf = 0.0;

    VectorArrayCoefficient dirichletBC(dim);
    Vector dispbc(pmesh->bdr_attributes.Max());
    dispbc = 0.0;                     // First attribute is homogeneous dirichlet boundary.
    dispbc(1) = (boundaryDisp[step]); // Second attribute is inhomogeneous dirichlet boundary.

    dirichletBC.Set(0, new PWConstCoefficient(dispbc));
    dirichletBC.Set(1, new ConstantCoefficient(0.0));
    if (dim == 3)
        dirichletBC.Set(2, new ConstantCoefficient(0.0));

    disp_gf.ProjectBdrCoefficient(dirichletBC, ess_bdr_x); // Ahh this is useless if we want to have different dirichlet boundaries for each component of displacement.

    // mfem::Vector pt{0.0, 0.0, 0.0};
    // int i = 0;
    // for (auto nd : ess_tdof_listx)
    // {
    //     pt(0) = (*node_coords)(nd);
    //     pt(1) = (*node_coords)(nd + num_nodes);
    //     if (dim == 3)
    //         pt(2) = (*node_coords)(nd + 2 * num_nodes);
    //     hat_u_x(i) = hat_u_x_func(pt, step);
    //     i++;
    // }
    // i = 0;
    // for (auto nd : ess_tdof_listy)
    // {
    //     nd -= num_nodes; // Offset by num_nodes for ess_tdof_listy
    //     pt(0) = (*node_coords)(nd);
    //     pt(1) = (*node_coords)(nd + num_nodes);
    //     if (dim == 3)
    //         pt(2) = (*node_coords)(nd + 2 * num_nodes);
    //     hat_u_y(i) = hat_u_y_func(pt, step);
    //     i++;
    // }
    // i = 0;
    // if (dim == 3)
    // {
    //     for (auto nd : ess_tdof_listz)
    //     {
    //         nd -= 2 * num_nodes; // Offset by 2 * num_nodes for ess_tdof_listz
    //         pt(0) = (*node_coords)(nd);
    //         pt(1) = (*node_coords)(nd + num_nodes);
    //         if (dim == 3)
    //             pt(2) = (*node_coords)(nd + 2 * num_nodes);
    //         hat_u_z(i) = hat_u_z_func(pt, step);
    //         i++;
    //     }
    // }
    // // Now set values in grid function.
    // {
    //     int i = 0;
    //     for (auto nd : ess_tdof_listx)
    //     {
    //         disp_gf(nd) = hat_u_x(i);
    //         i++;
    //     }

    //     i = 0;
    //     for (auto nd : ess_tdof_listy)
    //     {
    //         disp_gf(nd) = hat_u_y(i);
    //         i++;
    //     }

    //     if (dim == 3)
    //     {
    //         i = 0;
    //         for (auto nd : ess_tdof_listz)
    //         {
    //             disp_gf(nd) = hat_u_z(i);
    //             i++;
    //         }
    //     }
    // }
    return 0;
}

//--------------------------------------------------------------------------------------------------------------------------------------------
// Damage displacement solver for fracture methods defined here.
//--------------------------------------------------------------------------------------------------------------------------------------------
void PhaseFieldSolver::InitializeMatricesVectors()
{
    // Matrices
    KdispMat = new HypreParMatrix();
    KdmgMat = new HypreParMatrix();

    // Vectors
    u_new = new HypreParVector(dispfespace);

    d_vec2 = new HypreParVector(damagefespace);
    d_max = new HypreParVector(damagefespace);
    d_new = new HypreParVector(damagefespace);

    FdispVec = new HypreParVector(dispfespace);
    FdmgVec = new HypreParVector(damagefespace);

    // Initialize vectors
    (*u_new) = 0.0;
    (*d_vec2) = (*d_max) = (*d_new) = 0.0;
    (*FdispVec) = 0.0;
    (*FdmgVec) = 0.0;
}

void PhaseFieldSolver::InitializeSolvers()
{
    KdispPrec = new HypreBoomerAMG();
    KdmgPrec = new HypreBoomerAMG();

    KdispCG = new HyprePCG(MPI_COMM_WORLD);
    KdmgCG = new HyprePCG(MPI_COMM_WORLD);

    KdispPrec->SetPrintLevel(0);
    KdispPrec->SetRelaxType(6); // Symm. Gauss-Seidel
    KdispPrec->SetCycleType(1);

    KdmgPrec->SetPrintLevel(0);
    KdmgPrec->SetRelaxType(6); // Symm. Gauss-Seidel
    KdmgPrec->SetCycleType(1);

    KdispCG->SetTol(1e-12);
    KdispCG->SetMaxIter(1000);
    KdispCG->SetPrintLevel(0);

    KdmgCG->SetTol(1e-12);
    KdmgCG->SetMaxIter(1000);
    KdmgCG->SetPrintLevel(0);
}

void PhaseFieldSolver::AssembleKdispMatFdispVec()
{
    // displacement boundary conditions already projected onto u.
    kdisp->Update(dispfespace);
    kdisp->Assemble();
    kdisp->Finalize();

    fdisp->Assemble();

    kdisp->FormLinearSystem(ess_tdof_list, *u, *fdisp, *KdispMat, *u_new, *FdispVec);
};
void PhaseFieldSolver::ComputeDisp()
{
    int myid = Mpi::WorldRank();

    KdispPrec->SetOperator(*KdispMat);
    KdispCG->SetOperator(*KdispMat);
    KdispCG->SetPreconditioner(*KdispPrec);

    KdispCG->Mult(*FdispVec, *u_new);                 // u_new = KdispMat^{-1} (F_disp)
    kdisp->RecoverFEMSolution(*u_new, *FdispVec, *u); // Update grid function.
};

void PhaseFieldSolver::AssembleKdmg1Mat() // This method needs to be called only once.
{
    kdmg1->Assemble();
    kdmg1->Finalize();
    Kdmg1Mat = (kdmg1->ParallelAssemble()); // Reassign the pointer. Ownership transferred.

    // Additionally, since this method is only called once, will use this to initialize KdmgMat too.
    *KdmgMat = *(kdmg1->ParallelAssemble()); // Initialize object.
    *KdmgMat = 0.0;
};

void PhaseFieldSolver::AssembleKdmgMatFdmgVec() // Kdmg2Mat and FdmgVec depend on displacement u.
{
    kdmg2->Update(damagefespace);
    kdmg2->Assemble();
    kdmg2->Finalize();

    Kdmg2Mat = (kdmg2->ParallelAssemble()); // Reassign the pointer. Ownership transferred.

    KdmgMat = Add(1.0, *Kdmg1Mat, 1.0, *Kdmg2Mat);

    fdmg->Assemble();
    FdmgVec = fdmg->ParallelAssemble();
};

void PhaseFieldSolver::ComputeDamage()
{
    KdmgPrec->SetOperator(*KdmgMat);
    KdmgCG->SetOperator(*KdmgMat);
    KdmgCG->SetPreconditioner(*KdmgPrec);

    KdmgCG->Mult(*FdmgVec, *d_vec2); // d_vec2 = KdmgMat^{-1} (F_dmgVec)
    *d = *d_vec2;
};

double PhaseFieldSolver::ComputeDamageError()
{
    for (int n = 0; n < d_max->Size(); n++)
    {
        (*d_max)(n) = fmax((*d_vec2)(n), (*d_new)(n));
    }

    add(*d_max, -1.0, *d_new, *d_new);
    local_damage_error = d_new->Normlinf(); // This is a local norm. Need the global, i.e., find the max value across all partitions.

    MPI_Allreduce(&local_damage_error, &global_damage_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    *d_new = *d_max;
    *d = *d_new;

    return global_damage_error;
};

void PhaseFieldSolver::SuppressBoundaryDamage()
{
    auto [num_dofs, dim, num_nodes] = GetNodeInfo(node_coords);

    // This works. However, need to find the right condition. Which region can the damage be suppressed?
    mfem::Vector pt{0.0, 0.0, 0.0};
    for (int nd = 0; nd < num_nodes; nd++)
    {
        pt(0) = (*node_coords)(nd);
        pt(1) = (*node_coords)(nd + num_nodes);
        if (dim == 3)
            pt(2) = (*node_coords)(nd + 2 * num_nodes);

        if (pt(0) < 0.005 || pt(0) > (0.1 - 0.005))
            (*d)(nd) = 0.0;
    }

    d->GetTrueDofs(*d_vec2);
};

void PhaseFieldSolver::ComputeHistoryVariable()
{
    // Compute history variable for each element, not for each node.
    // Also compute strain energy for each element, not for each node. At least for now.
    // Fill up an L2 grid function.
    mfemplus::AccessMFEMFunctions accessFunc;

    int num_elements = damagefespace->GetNE();
    auto [num_dofs, dim, num_nodes] = GetNodeInfo(node_coords);
    int dof, str_comp;
    str_comp = (dim == 2) ? 3 : 6;

    Vector eldofdisp;
    DenseMatrix dshape, gshape;
    Array<int> eldofs;

    DenseMatrix C, B, CB; // stiffness, strain-displacement, Stiffness times strain-displacement in Voigt form
    Vector CBu, Bu;

    for (int elnum = 0; elnum < num_elements; elnum++)
    {
        const FiniteElement &el = *(damagefespace->GetFE(elnum));
        ElementTransformation &Trans = *(damagefespace->GetElementTransformation(elnum));

        dof = el.GetDof();

        if (elnum == 0)
        {
            dshape.SetSize(dof, dim);
            gshape.SetSize(dof, dim);

            eldofs.SetSize(dof * dim);    // vector valued for displacement
            eldofdisp.SetSize(dof * dim); // vector valued displacement
        }

        dispfespace->GetElementVDofs(elnum, eldofs);

        for (int i = 0; i < eldofdisp.Size(); i++)
        {
            eldofdisp(i) = (*d)(eldofs[i]);
        }
        // Great, now we have all components of displacements at each dof.
        // Next, construct the stiffness matrix C, compute displacement gradients, and take inner product.

        const mfem::IntegrationRule *ir = accessFunc.GetIntegrationRule(el, Trans);

        if (elnum == 0)
        {
            C.SetSize(str_comp, str_comp);
            B.SetSize(str_comp, dof * dim);
            CB.SetSize(str_comp, dof * dim);
            CBu.SetSize(str_comp);
            Bu.SetSize(str_comp);
        }

        double NU, E, w;
        double strain_energy(0.0), temp(0.0);
        w = 1.0 / ir->GetNPoints(); // To take average of strain energy in the element.

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);

            el.CalcDShape(ip, dshape);
            Trans.SetIntPoint(&ip);

            // need to give coefficients to evaluate history variable.
            // NU = poisson_ratio->Eval(Trans, ip);
            // E = young_mod->Eval(Trans, ip); // The elastic constants are evaluated at each integration point.

            mfem::Mult(dshape, Trans.InverseJacobian(), gshape); // Recovering the gradients of the shape functions in the physical space.

            // Here we want to use Voigt notation to speed up the assembly process.
            // For this, we need the strain displacement matrix B. The element stiffness can be computed as
            // \int_{\Omega} B^T C B. In Voigt form, the stiffness matrix has dimensions 3 x 3 in 2D and 6 x 6 in 3D.
            // The B matrix as 3 rows in 2D and 6 rowd in 3D.

            // add(elvect, ip.weight * val, shape, elvect);
            if (dim == 2)
            {
                C = 0.0;
                // Plane strain
                C(0, 0) = C(1, 1) = E * (1 - NU) / ((1 + NU) * (1 - 2 * NU));
                C(0, 1) = C(1, 0) = E * NU / ((1 + NU) * (1 - 2 * NU));
                C(2, 2) = E / (2 * (1 + NU));

                // Plane stress
                // C(0, 0) = C(1, 1) = (E / (1 - pow(NU, 2)));
                // C(0, 1) = C(1, 0) = (E * NU / (1 - pow(NU, 2)));
                // C(2, 2) = (E * (1 - NU) / (2 * (1 - pow(NU, 2))));

                // In 2D, we have 3 unique strain components.
                B = 0.0;
                for (int spf = 0; spf < dof; spf++)
                {
                    B(0, spf) = gshape(spf, 0);
                    B(1, spf + dof) = gshape(spf, 1);
                    B(2, spf) = gshape(spf, 1);
                    B(2, spf + dof) = gshape(spf, 0);
                }
            }

            else if (dim == 3)
            {
                C = 0.0;
                C(0, 0) = C(1, 1) = C(2, 2) = (E * (1 - NU)) / ((1 - 2 * NU) * (1 + NU));
                C(0, 1) = C(0, 2) = C(1, 0) = C(1, 2) = C(2, 0) = C(2, 1) = (E * NU) / ((1 - 2 * NU) * (1 + NU));
                C(3, 3) = C(4, 4) = C(5, 5) = E / (2 * (1 + NU));

                // In 3D, we have 6 unique strain components.
                B = 0.0;
                for (int spf = 0; spf < dof; spf++)
                {
                    B(0, spf) = gshape(spf, 0);
                    B(1, spf + dof) = gshape(spf, 1);
                    B(2, spf + 2 * dof) = gshape(spf, 2);
                    B(3, spf + dof) = gshape(spf, 2);
                    B(3, spf + 2 * dof) = gshape(spf, 1);
                    B(4, spf) = gshape(spf, 2);
                    B(4, spf + 2 * dof) = gshape(spf, 0);
                    B(5, spf) = gshape(spf, 1);
                    B(5, spf + dof) = gshape(spf, 0);
                }
            }

            // Now compute the quantity C_{ijkl} u_{k,l} u_{i,j}. Using Voigt notation, of course...
            // This is equivalent to.
            mfem::Mult(C, B, CB);    // CB is 6 x (dof * dim)
            CB.Mult(eldofdisp, CBu); // CBu has dimension strain_comps. This is the stress vector.
            B.Mult(eldofdisp, Bu);   // Bu has dimension strain_comps. This is the strain vector.
            temp = mfem::InnerProduct(CBu, Bu);
            strain_energy += w * temp;
        }
        // Here, we need to check if old strain_energy, i.e. the magnitude of the number already stored by h is greater or less than the newly computed strain energy for this element. If the old one is greater, don't update. If new one is greater, update.
        if (strain_energy > (*h)(elnum))
        {
            (*h)(elnum) = strain_energy;
        } // Only updates if newly computed strain energy is greater.
    };
    // This history grid function can be used in the bilinear and linear form integrators.
};