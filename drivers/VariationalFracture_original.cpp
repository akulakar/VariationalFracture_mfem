// -----------------------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------------------
// ---------------------------------- Variational Fracture Code ----------------------------------
// Algorithm will be written here shortly.
// -----------------------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------------------

#include <iostream>
#include <cmath>
#include <string>
#include <mfem.hpp>
#include <mfemplus.hpp>
#include <nlohmann/json.hpp>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace mfem;
using namespace std;
using json = nlohmann::json;

struct simProps
{
    double t_final; // final step
    double dt_inc;  // step size
    double dt_save; // save increment
    int N;          // Number of steps
};

std::tuple<int, int, int> GetNodeInfo(const mfem::GridFunction *nodes_ptr);
bool readInputParameters(int argc, char *argv[], json &inputParameters);

class BoundaryConditions
{

private:
    json inputFile;
    mfem::ParMesh *pmesh;
    mfem::ParFiniteElementSpace *pfespace;
    mfem::Array<int> ess_bdr_x, ess_bdr_y, ess_bdr_z;
    mfem::Array<int> ess_tdof_list_x, ess_tdof_list_y, ess_tdof_list_z;
    const mfem::GridFunction *node_coords;

protected:
    double hat_u_x_func(const mfem::Vector &pt, double &time);
    double hat_u_y_func(const mfem::Vector &pt, double &time);
    double hat_u_z_func(const mfem::Vector &pt, double &time);

public:
    BoundaryConditions(const json &inputParameters, mfem::ParMesh *mesh_in, mfem::ParFiniteElementSpace *fespace_bc) : inputFile(inputParameters), pmesh(mesh_in), pfespace(fespace_bc)
    {
        node_coords = pmesh->GetNodes();
    };

    int determineDirichletDof(mfem::Array<int> &ess_tdof_listx, mfem::Array<int> &ess_tdof_listy, mfem::Array<int> &ess_tdof_listz, mfem::ParGridFunction &hat_u_x_nodes, mfem::ParGridFunction &hat_u_y_nodes, mfem::ParGridFunction &hat_u_z_nodes);
    int createDirichletVals(mfem::Array<int> &ess_tdof_listx, mfem::Array<int> &ess_tdof_listy, mfem::Array<int> &ess_tdof_listz, mfem::Vector &hat_u_x, mfem::Vector &hat_u_y, mfem::Vector &hat_u_z, double time);
};

class PhaseFieldSolver
{
protected:
    json inputFile;
    Array<int> ess_tdof_list;
    const mfem::GridFunction *node_coords;
    ParGridFunction *u, *d;
    ParMesh *pmesh;
    ParFiniteElementSpace *dispfespace, *damagefespace;
    ParBilinearForm *kdisp, *kdmg1, *kdmg2;
    ParLinearForm *fdisp, *fdmg;
    HypreParMatrix *KdispMat, *KdmgMat, *Kdmg1Mat, *Kdmg2Mat;
    HypreParVector *FdispVec, *FdmgVec;
    HypreParVector *u_old, *u_new, *d_vec2, *d_max, *d_new;
    HypreBoomerAMG *KdispPrec, *KdmgPrec;
    HyprePCG *KdispCG, *KdmgCG;
    double local_damage_error, global_damage_error, error_tolerance;

public:
    PhaseFieldSolver(double error_tol) : error_tolerance(error_tol) {};
    void ReadFESpacesMeshEssDofs(ParFiniteElementSpace *disp_fes, ParFiniteElementSpace *damage_fes, ParMesh *pmesh_in, Array<int> &ess_dofs)
    {
        dispfespace = disp_fes;
        damagefespace = damage_fes;
        pmesh = pmesh_in;
        ess_tdof_list = ess_dofs;
        node_coords = pmesh->GetNodes();
    };
    // Read bilinear and linear forms.
    void ReadBilinearLinearForms(ParBilinearForm *k_disp, ParBilinearForm *k_damage1, ParBilinearForm *k_damage2, ParLinearForm *f_disp, ParLinearForm *f_damage = nullptr)
    {
        kdisp = k_disp;
        kdmg1 = k_damage1;
        kdmg2 = k_damage2;
        fdisp = f_disp;
        fdmg = f_damage; // If no traction or body forces, this pointer is NULL.
    };
    // Read grid functions.
    void ReadGridFunctions(ParGridFunction &disp_gf, ParGridFunction &damage_gf)
    {
        u = &disp_gf;
        d = &damage_gf;
    };

    void InitializeMatricesVectors(); // Initializes HypreParMatrices for bilinar forms. Initializes HypreParVectors for linear forms.
    void InitializeSolvers();
    void ProjectDisplacementBCs();
    void AssembleKdispMatFdispVec();
    void ComputeDisp();
    void AssembleKdmg1Mat();
    void AssembleKdmg2MatFdmgVec();
    void ComputeDamage();
    double ComputeDamageError();
    void UpdateGridFunctions();
    ~PhaseFieldSolver()
    {
        delete KdispMat;
        delete Kdmg1Mat;
        delete Kdmg2Mat;
        delete FdispVec;
        delete FdmgVec;
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

    StopWatch total_time, solver_time, postprocessing_time;
    total_time.Start();

    // Initialize MPI and HYPRE.
    Mpi::Init(argc, argv);
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();
    Device device("cpu");
    // Print OpenMP threads

    json inputParameters;
    simProps fractureSimProps;
    readInputParameters(argc, argv, inputParameters); // read on every MPI rank.

    fractureSimProps.N = 10;

    std::string mesh_file_str = inputParameters["Simulation Parameters"]["Mesh Parameters"]["meshFileName"];
    const char *mesh_file = mesh_file_str.c_str(); // open mesh file on every MPI rank. Create parallel mesh and delete mesh.
    int order = inputParameters["Simulation Parameters"]["Mesh Parameters"]["order"];
    int ref_levels = inputParameters["Simulation Parameters"]["Mesh Parameters"]["ref_levels"];
    // Add other relevant parameters.

    //------------------------------------------------------------------------------------------------------------------
    // Read the mesh from the given mesh file.
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();      // dim should equal 3.
    mesh->SetCurvature(order, false); // Enable high-order geometry
    for (int l = 0; l < ref_levels; l++)
        mesh->UniformRefinement();

    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

    double lambda = 2.0;
    double mu = 1.0;

    //------------------------------------------------------------------------------------------------------------------
    // Define finite element spaces.
    // We are using elements close to Taylor-Hood elements.
    // P_{k} vector Lagrange elements for displacement and P_{k} scalar lagrage elements for damage;

    FiniteElementCollection *Lagrangefec(new H1_FECollection(order, dim, BasisType::GaussLobatto));
    ParFiniteElementSpace *dispfespace, *damagefespace;

    dispfespace = new ParFiniteElementSpace(pmesh, Lagrangefec, dim); // Define a vector H1 finite element space for nodal displacements.
    damagefespace = new ParFiniteElementSpace(pmesh, Lagrangefec, 1); // Define a scalar H1 finite element space for nodal pressures.
    pmesh->SetNodalFESpace(dispfespace);

    // Define L2 finite element spaces for stress.

    //------------------------------------------------------------------------------------------------------------------
    // Define and initialize all grid functions

    ParGridFunction u(dispfespace), d(damagefespace);
    u = 0.0;
    d = 0.0;

    // damagefespace is a scalar version of dispfespace, so will use it as the fespace for BCs.
    ParGridFunction hat_u_x_nodes(damagefespace), hat_u_y_nodes(damagefespace), hat_u_z_nodes(damagefespace);
    hat_u_x_nodes = 0.0;
    hat_u_y_nodes = 0.0;
    hat_u_z_nodes = 0.0;

    //------------------------------------------------------------------------------------------------------------------

    pmesh->EnsureNodes();
    GridFunction *nodes = pmesh->GetNodes();
    Array<int> glob_elem_ids;
    pmesh->GetGlobalElementIndices(glob_elem_ids);
    const int vdim = nodes->VectorDim(); // Should be 3 for 2D
    const int num_nodes = nodes->Size() / vdim;
    const int num_els = dispfespace->GetNE();

    //------------------------------------------------------------------------------------------------------------------
    // Read boundary attributes and determine essential dofs in each parition.
    BoundaryConditions BCs(inputParameters, pmesh, damagefespace);

    Array<int> ess_bdr_x(pmesh->bdr_attributes.Max()), ess_bdr_y(pmesh->bdr_attributes.Max()), ess_bdr_z(pmesh->bdr_attributes.Max());
    Array<int> ess_tdof_listbcx, ess_tdof_listbcy, ess_tdof_listbcz, ess_tdof_list;

    // BCs.determineDirichletDof(ess_tdof_listbcx, ess_tdof_listbcy, ess_tdof_listbcz, hat_u_x_nodes, hat_u_y_nodes, hat_u_z_nodes);
    // ess_tdof_list.Append(ess_tdof_listbcx);
    // ess_tdof_list.Append(ess_tdof_listbcy);
    // ess_tdof_list.Append(ess_tdof_listbcz);
    Vector hat_u_x(ess_tdof_listbcx.Size()), hat_u_y(ess_tdof_listbcy.Size()), hat_u_z(ess_tdof_listbcz.Size());
    // BCs.createDirichletVals(ess_tdof_listbcx, ess_tdof_listbcy, ess_tdof_listbcz, hat_u_x, hat_u_y, hat_u_z, 0);

    // Project dirichlet values onto displacement grid function

    //------------------------------------------------------------------------------------------------------------------
    if (myid == 0)
        cout << "Assembling bilinear and linear forms" << endl;

    // Stiffness matrix for displacement (K_{u}) is the elastic strain energy multiplied by degradation function (1 - d^{2}) + k_{\varepsilon}
    // Stiffness matrix and corresponding boundary dislacements load term is recomputed at every iteration and sub iteration.

    // Stiffness matrix 1 for damage (K_{d1}) is computed only once per iteration (not during sub iterations);
    // Stiffness matrix 2 for damage (K_{d2}) is computed every sub iteration.
    // Load vector for damage (F_{d}) is computed every sub iteration.

    // In total:
    // 1 bilinear form for displacement, possibly 1 linear form for displacement if growth term is included.
    // 2 bilinear forms and 1 linear for for damage.
    // Set up bilinear forms with integrators and pass pointers to the solver class. In the class the bilinear forms will be assembled as required.

    // Define bilinear and mixed forms.
    // Stiffness matrix for displacement term
    // Need a coefficient value for each computational node. Stiffness matrix depends on damage d at each node... (or element?)
    // But quadrature points are not the same as nodal points... So need to construct an interpolant in each element with shape functions and evaluate damage at each quadrature point...

    // Kdisp
    double Ey = mu * (3 * lambda + 2 * mu) / (lambda + mu);
    double nu = lambda / (2 * (lambda + mu));
    double Keps = 1.0e-7;
    ConstantCoefficient Ey_coeff(Ey), nu_coeff(nu);

    ParBilinearForm *kdisp(new ParBilinearForm(dispfespace));
    kdisp->AddDomainIntegrator(new mfemplus::IsotropicElasticityDamageIntegrator(Ey_coeff, nu_coeff, Keps, d, damagefespace));
    // kdisp->Assemble();
    // Looks good for now.

    // Kdmg1
    double g = 0.4;
    double epsilon = 0.2;
    ConstantCoefficient g_over_eps_coeff(g / epsilon), g_times_eps_coeff(g * epsilon);

    ParBilinearForm *kdmg1(new ParBilinearForm(damagefespace));
    kdmg1->AddDomainIntegrator(new MassIntegrator(g_over_eps_coeff));
    kdmg1->AddDomainIntegrator(new DiffusionIntegrator(g_times_eps_coeff));
    // kdmg1->Assemble();
    // Looks good. This term is simple.

    // Kdmg2
    // Need to compute the stress : strain inner product at each quad point.
    ParBilinearForm *kdmg2(new ParBilinearForm(damagefespace));
    kdmg2->AddDomainIntegrator(new mfemplus::IsotropicStrainEnergyDamageIntegrator(Ey_coeff, nu_coeff, u, dispfespace));
    // kdmg2->Assemble();
    // Looks good for now.

    // Linear forms

    // fdisp
    ParLinearForm *fdisp(new ParLinearForm(dispfespace));
    // fdisp->Assemble(); // 0 if no body forces.

    // fdmg
    // Need to compute the stress :: strain inner product for this too at each quad point.
    ParLinearForm *fdmg(new ParLinearForm(damagefespace));
    fdmg->AddDomainIntegrator(new mfemplus::FractureDamageLFIntegrator(Ey_coeff, nu_coeff, u, dispfespace)); // Wrong coeff, need to change.
    // fdmg->Assemble();
    // Looks good for now.

    // Set up object to save data

    std::string resultsFolder = "../results/" + inputParameters["testName"].get<std::string>();

    adios2stream aos(resultsFolder + "/WavesMFEMoutput.bp", adios2stream::openmode::out, MPI_COMM_WORLD, "BP5");
    {
        aos.SetRefinementLevel(0);
        aos.SetParameter("FlushStepsCount", "5");
        aos.BeginStep();
        pmesh->Print(aos);
        // Write fields
        u.Save(aos, "disp", adios2stream::data_type::point_data);
        d.Save(aos, "damage", adios2stream::data_type::point_data);
        hat_u_x_nodes.Save(aos, "hat_u_x_nodes", adios2stream::data_type::point_data);
        hat_u_x_nodes.Save(aos, "hat_u_y_nodes", adios2stream::data_type::point_data);
        aos.EndStep();
    }
    //------------------------------------------------------------------------------------------------------------------
    // Initialize the main fracture class.

    double error_tol = 1e-5;

    PhaseFieldSolver PhaseFieldFractureSolver(error_tol);

    PhaseFieldFractureSolver.ReadFESpacesMeshEssDofs(dispfespace, damagefespace, pmesh, ess_tdof_list);
    PhaseFieldFractureSolver.ReadBilinearLinearForms(kdisp, kdmg1, kdmg2, fdisp, fdmg);
    PhaseFieldFractureSolver.ReadGridFunctions(u, d);
    PhaseFieldFractureSolver.InitializeMatricesVectors();
    PhaseFieldFractureSolver.InitializeSolvers();

    double damage_error = 1.0; // Initializing.
    // double time = fractureSimProps.dt_inc;
    int max_iterations = 100;
    int iterations = 0;

    for (int step = 1; step <= fractureSimProps.N; step++)
    {
        if (step == 1)
        {
            // Kdmg1Mat is only assembled once during the entire simulation. Slightly weird...
            PhaseFieldFractureSolver.AssembleKdmg1Mat();
        }

        // BCs.createDirichletVals(ess_tdof_listbcx, ess_tdof_listbcy, ess_tdof_listbcz, hat_u_x, hat_u_y, hat_u_z, step);

        damage_error = 1.0;
        iterations = 0;
        while (damage_error >= error_tol && iterations <= max_iterations)
        {

            PhaseFieldFractureSolver.ProjectDisplacementBCs();
            PhaseFieldFractureSolver.AssembleKdispMatFdispVec();
            PhaseFieldFractureSolver.ComputeDisp();
            // Computed displacements.

            PhaseFieldFractureSolver.AssembleKdmg2MatFdmgVec();
            PhaseFieldFractureSolver.ComputeDamage();
            // Now need to compute the max damage at each computational node.
            damage_error = PhaseFieldFractureSolver.ComputeDamageError();
        };

        if (iterations >= max_iterations && myid == 0)
        {
            cout << "Failed. Damage field did not converge!" << endl;
            // kill program.
            std::error_code ec;
            std::cerr << "Failed. Damage field did not converge!" << ec.message() << std::endl;
            _exit(2);
        }
    }

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

//----------------------------------------------------------------------------------------------------------------------------------------------
// BoundaryConditions methods defined here.
//----------------------------------------------------------------------------------------------------------------------------------------------
int BoundaryConditions::determineDirichletDof(mfem::Array<int> &ess_tdof_listx, mfem::Array<int> &ess_tdof_listy, mfem::Array<int> &ess_tdof_listz, mfem::ParGridFunction &hat_u_x_nodes, mfem::ParGridFunction &hat_u_y_nodes, mfem::ParGridFunction &hat_u_z_nodes)
{
    int myid = Mpi::WorldRank();

    mfem::Array<int> bdr_nodes_list;
    // pfespace->GetBoundaryTrueDofs(bdr_nodes_list, 0); // This didn't work, so doing roundabout way instead.

    mfem::Array<int> all_bdr_attr_selector(pmesh->bdr_attributes.Max());
    // mfem::Array<int> all_bdr_attr_selector(1);
    // First attribute is for the dirichlet nodes.
    all_bdr_attr_selector = 0;
    all_bdr_attr_selector[0] = 1; // only first attribute is dirichlet...
    // all_bdr_attr_selector = 1;
    GridFunction nodes(pfespace);
    nodes = 0.0;
    ConstantCoefficient one(1.0);
    nodes.ProjectBdrCoefficient(one, all_bdr_attr_selector);

    for (int i = 0; i < nodes.Size(); i++)
        if (nodes(i) != 0.0)
            bdr_nodes_list.Append(i);

    auto [num_dofs, vdim, num_nodes] = GetNodeInfo(node_coords);

    // MFEM_VERIFY(nodes, "Mesh has no nodes. Did you call SetCurvature or SetNodalFESpace?");

    double max_coord = 0;
    double temp;
    for (auto nd : bdr_nodes_list)
    {
        temp = (*node_coords)(nd);
        max_coord = std::max(max_coord, abs(temp));
    }
    max_coord -= 0.0008;

    for (auto nd : bdr_nodes_list)
    {
        mfem::Vector pt(3);
        pt(0) = (*node_coords)(nd);
        pt(1) = (*node_coords)(nd + num_nodes);
        pt(2) = (*node_coords)(nd + 2 * num_nodes);

        // All nodes in this atribute are Dirichlet nodes. No constraint on coordinates.
        // All three vector dofs are fixed.

        // if ((pt(1) <= -0.0784) && (std::abs(pt(0)) <= 0.037) && (std::abs(pt(2)) <= 0.037))
        // if ((pt(0) * pt(0)) + (pt(1) * pt(1)) + (pt(2) * pt(2)) >= (max_coord * max_coord))
        {
            ess_tdof_listx.Append(nd);
            hat_u_x_nodes(nd) = 1.0;

            ess_tdof_listy.Append(nd);
            hat_u_y_nodes(nd) = 1.0;

            ess_tdof_listz.Append(nd);
            hat_u_z_nodes(nd) = 1.0;
        }
    }
    return 0;
}

// Boudary conditions need to be a function of time.
double BoundaryConditions::hat_u_x_func(const mfem::Vector &pt, double &time)
{
    // double x1 = pt(0);
    // double x2 = pt(1);
    // double x3 = pt(2);
    return 0.0; // Homogeneous Dirichlet condition
}

double BoundaryConditions::hat_u_y_func(const mfem::Vector &pt, double &time)
{
    // double x1 = pt(0);
    // double x2 = pt(1);
    // double x3 = pt(2);
    return 0.0; // Homogeneous Dirichlet condition
}

double BoundaryConditions::hat_u_z_func(const mfem::Vector &pt, double &time)
{
    // double x1 = pt(0);
    // double x2 = pt(1);
    // double x3 = pt(2);
    return 0.0; // Homogeneous Dirichlet condition
}

int BoundaryConditions::createDirichletVals(mfem::Array<int> &ess_tdof_listx, mfem::Array<int> &ess_tdof_listy, mfem::Array<int> &ess_tdof_listz, mfem::Vector &hat_u_x, mfem::Vector &hat_u_y, mfem::Vector &hat_u_z, double time)
{
    auto [num_dofs, vdim, num_nodes] = GetNodeInfo(node_coords);
    mfem::Vector pt{0.0, 0.0, 0.0};
    int i = 0;
    for (auto nd : ess_tdof_listx)
    {

        pt[0] = (*node_coords)(nd);
        pt[1] = (*node_coords)(nd + num_nodes);
        pt[2] = (*node_coords)(nd + 2 * num_nodes);
        hat_u_x(i) = hat_u_x_func(pt, time);
        i++;
    }
    i = 0;
    for (auto nd : ess_tdof_listy)
    {

        pt[0] = (*node_coords)(nd);
        pt[1] = (*node_coords)(nd + num_nodes);
        pt[2] = (*node_coords)(nd + 2 * num_nodes);
        hat_u_y(i) = hat_u_y_func(pt, time);
        i++;
    }
    i = 0;
    for (auto nd : ess_tdof_listz)
    {

        pt[0] = (*node_coords)(nd);
        pt[1] = (*node_coords)(nd + num_nodes);
        pt[2] = (*node_coords)(nd + 2 * num_nodes);
        hat_u_z(i) = hat_u_z_func(pt, time);
        i++;
    }

    return 0;
}

//
//
// Damage displacement solver for fracture methods defined here.
void PhaseFieldSolver::InitializeMatricesVectors()
{
    u_old = new HypreParVector(dispfespace);
    u_new = new HypreParVector(dispfespace);

    d_vec2 = new HypreParVector(damagefespace);
    d_max = new HypreParVector(damagefespace);
    d_new = new HypreParVector(damagefespace);

    // Initialize vectors
    (*u_old) = (*u_new) = 0.0;
    (*d_vec2) = (*d_max) = (*d_new) = 0.0;
}

void PhaseFieldSolver::InitializeSolvers()
{
    // This may not work because the pointer is null, but it initializes by pointer...
    // Maybe the better way to do this is to set up the first initialization outside the class in the main script and then pass it to the class.
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

    KdispCG->SetTol(1e-10);
    KdispCG->SetMaxIter(1000);
    KdispCG->SetPrintLevel(0);
    KdispCG->SetPreconditioner(*KdispPrec);

    KdmgCG->SetTol(1e-10);
    KdmgCG->SetMaxIter(1000);
    KdmgCG->SetPrintLevel(0);
    KdmgCG->SetPreconditioner(*KdmgPrec);
}

void PhaseFieldSolver::ProjectDisplacementBCs() {};

void PhaseFieldSolver::AssembleKdispMatFdispVec()
{
    // displacement boundary conditions already projected onto u.
    kdisp->Assemble();
    kdisp->Finalize();
    // Currently this line doesn't work due to "MFEM abort: Unknown host memory controller!"
    kdisp->FormLinearSystem(ess_tdof_list, *u, *fdisp, *KdispMat, *u_new, *FdispVec);
};
void PhaseFieldSolver::ComputeDisp()
{
    KdispPrec->SetOperator(*KdispMat);
    KdispCG->SetOperator(*KdispMat);
    KdispCG->SetPreconditioner(*KdispPrec);

    KdispCG->Mult(*FdispVec, *u_new); // u_new = KdispMat^{-1} (F_disp)
};

void PhaseFieldSolver::AssembleKdmg1Mat() // This method need not be called every sub iteration.
{
    kdmg1->Assemble();
    kdmg1->Finalize();
    Kdmg1Mat = (kdmg1->ParallelAssemble()); // Reassign the pointer. Ownership transferred.
};

void PhaseFieldSolver::AssembleKdmg2MatFdmgVec() // Kdmg2Mat and FdmgVec depend on displacement u.
{
    kdmg2->Assemble();
    kdmg2->Finalize();
    Kdmg2Mat = (kdmg2->ParallelAssemble()); // Reassign the pointer. Ownership transferred.

    fdmg->Assemble();
    FdmgVec = fdmg->ParallelAssemble();
};

void PhaseFieldSolver::ComputeDamage()
{
    KdmgMat->Add(1.0, *Kdmg1Mat);
    KdmgMat->Add(1.0, *Kdmg2Mat);

    KdmgPrec->SetOperator(*KdmgMat);
    KdmgCG->SetOperator(*KdmgMat);
    KdmgCG->SetPreconditioner(*KdmgPrec);

    KdmgCG->Mult(*FdmgVec, *d_vec2); // d_new = KdmgMat^{-1} (F_dmg)
};

double PhaseFieldSolver::ComputeDamageError()
{

    for (int n = 0; n < d_vec2->Size(); n++)
    {
        (*d_max)(n) = fmax((*d_vec2)(n), (*d_new)(n));
    }

    add(*d_max, -1.0, *d_new, *d_new);
    local_damage_error = d_new->Normlinf(); // This is a local norm. Need the global, i.e., find the max value across all partitions.

    MPI_Allreduce(&local_damage_error, &global_damage_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    *d_new = *d_max;

    return global_damage_error;
};

void PhaseFieldSolver::UpdateGridFunctions()
{
    // Grid functions are set to hypreparvector values.
    *u = *u_new;
    *d = *d_new;
};