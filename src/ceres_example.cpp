#include <sstream>
#include <fstream>
#include <iostream>
#include <random>
#include <memory>

#include <Eigen/Core>
#include <gflags/gflags.h>
#include <ceres/ceres.h>
#include <glog/logging.h>

/** Sample functor which simply computes the residual.
 *  Used for numeric differentiation.
 */
struct NumericEllipseCostFunctor
{
    NumericEllipseCostFunctor(const Eigen::Vector2d &observed_point) : observed_point(observed_point) {}
    bool operator()(const double *const parameters, double *residuals) const
    {
        // extract Ellipse params
        const double h = parameters[0];
        const double k = parameters[1];
        const double a = parameters[2];
        const double b = parameters[3];

        // compute the cost
        double dx = observed_point.x() - h;
        double dy = observed_point.y() - k;
        residuals[0] = (dx * dx) / (a * a) + (dy * dy) / (b * b) - 1;
        return true;
    }

    Eigen::Vector2d observed_point;
};

/** Slightly more advanced functor which is templated, allowing
 *  Ceres to automatically compute the Jacobian using
 *  templates.
 */
struct AutoEllipseCostFunctor
{
    AutoEllipseCostFunctor(const Eigen::Vector2d &observed_point) : observed_point(observed_point) {}

    /** Ceres will create a version of this with a special
     *  autodiff type for determining the Jacobian
     *  and another with doubles for residual computation
     */
    template <typename T>
    bool operator()(const T *const parameters, T *residuals) const
    {
        const T h = parameters[0];
        const T k = parameters[1];
        const T a = parameters[2];
        const T b = parameters[3];
        const T x = T(observed_point.x());
        const T y = T(observed_point.y());

        T dx = T(observed_point.x()) - h;
        T dy = T(observed_point.y()) - k;
        residuals[0] = (dx * dx) / (a * a) + (dy * dy) / (b * b) - T(1.0);
        return true;
    }

    Eigen::Vector2d observed_point;
};

/** If the analytic gradient is simple to compute or if perfomance is a concern,
 *  it can be best to compute the Jacobians by hand, as shown here. The template
 *  arguments indicate to Ceres the number of residuals, and the number of
 *  parameters. This can also be determined dynamically using the base
 *  class `CostFunction`.
 */
struct AnalyticEllipseCostFunction : public ceres::SizedCostFunction<1, 4>
{
    AnalyticEllipseCostFunction(const Eigen::Vector2d &observed_point) : observed_point(observed_point) {}
    virtual ~AnalyticEllipseCostFunction() {}

    /** This function performs double duty: it both computes the residuals and,
     *  at other times, will also compute the Jacobians. This is communicated
     *  via potential `nullptr` values in `jacobians`. While somewhat awkward,
     *  this allows for re-use of sub-expressions for increased efficiency.
     *  The sizes of the arrays are indicated via the template argument
     *  above.
     *
     *  \param parameters an array of parameter arrays
     *  \param residuals an array of residuals values
     *  \param jacobians an array of Jacobian matrices. Each matrix is in row-major order.
     *  \return whether the evaluation was successful
     */
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        const double h = parameters[0][0];
        const double k = parameters[0][1];
        const double a = parameters[0][2];
        const double b = parameters[0][3];

        // We can re-use all of these later
        const double dx = observed_point.x() - h;
        const double dy = observed_point.y() - k;
        const double dx2 = dx * dx;
        const double dy2 = dy * dy;
        const double a2 = a * a;
        const double b2 = b * b;
        residuals[0] = dx2 / a2 + dy2 / b2 - 1;

        // will be null if only evaluating residuals
        if (jacobians != nullptr)
        {
            // if some parameters are being held constant,
            // then individual Jacobian matrices will also be null
            // to avoid unneeded computation
            if (jacobians[0] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> jac(jacobians[0]);
                jac(0, 0) = (-2 * dx) / a2;
                jac(0, 1) = (-2 * dy) / b2;
                jac(0, 2) = (-2 * dx2) / (a2 * a);
                jac(0, 3) = (-2 * dy2) / (b2 * b);
            }
        }

        return true;
    }

    Eigen::Vector2d observed_point;
};

/** We can inject our own code into the optimization process to do
 *  custom logging and the like. This class writes intermediate values
 *  to a CSV file.
 */
class CSVCallback : public ceres::IterationCallback
{
public:
    explicit CSVCallback(const std::string &path, const double *params, int num_observations)
        : m_params(params), m_num_observations(num_observations), m_output(path)
    {
        m_output << "Cost,h,k,a,b" << std::endl;
    }

    ~CSVCallback() {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary)
    {
        m_output << summary.cost / m_num_observations << "," << m_params[0] << "," << m_params[1] << "," << m_params[2] << "," << m_params[3] << std::endl;
        return ceres::CallbackReturnType::SOLVER_CONTINUE;
    }

private:
    const double *m_params;
    const int m_num_observations;
    std::ofstream m_output;
};

/** Creates a dataset consisting of noisy samples from an arc of an
 *  axis-aligned ellipse.
 * 
 *  \param num_observations the number of observations to sample
 *  \param params the ellipse parameters
 *  \param start_angle the starting angle of the arc in radians
 *  \param end_angle the ending angle of the arc in radians
 *  \param noise_sigma the sigma of the Gaussian used for noise
 *  \return a matrix of points
 */
Eigen::Matrix2Xd create_dataset(int num_observations,
                                const double *params,
                                double start_angle = -0.5,
                                double end_angle = 2.0,
                                double noise_sigma = 0.05)
{
    const double h = params[0];
    const double k = params[1];
    const double a = params[2];
    const double b = params[3];
    Eigen::RowVectorXd angles = Eigen::RowVectorXd::LinSpaced(num_observations, start_angle, end_angle);
    Eigen::Matrix2Xd data(2, num_observations);
    data.row(0) = (a * angles.array().cos()) + h;
    data.row(1) = (b * angles.array().sin()) + k;
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, noise_sigma};
    for (auto i = 0; i < data.cols(); ++i)
    {
        data(0, i) += d(gen);
        data(1, i) += d(gen);
    }

    return data;
}

// Setup the problem using numeric differentiation.
void setup_numeric(ceres::Problem &problem, const Eigen::Matrix2Xd &dataset, double *params)
{
    std::cout << "Numeric differentiation: " << std::endl;
    for (auto i = 0; i < dataset.cols(); ++i)
    {
        ceres::CostFunction *cost_function =
            new ceres::NumericDiffCostFunction<NumericEllipseCostFunctor, ceres::CENTRAL, 1, 4>(
                new NumericEllipseCostFunctor(dataset.col(i)));
        problem.AddResidualBlock(cost_function, nullptr, params);
    }
}

// Setup the problem using automatic differentiation.
void setup_autodiff(ceres::Problem &problem, const Eigen::Matrix2Xd &dataset, double *params)
{
    std::cout << "Automatic differentiation: " << std::endl;
    for (auto i = 0; i < dataset.cols(); ++i)
    {
        ceres::CostFunction *cost_function =
            new ceres::AutoDiffCostFunction<AutoEllipseCostFunctor, 1, 4>(
                new AutoEllipseCostFunctor(dataset.col(i)));
        problem.AddResidualBlock(cost_function, nullptr, params);
    }
}

// Setup the problem using analytic differentiation.
void setup_analytic(ceres::Problem &problem, const Eigen::Matrix2Xd &dataset, double *params)
{
    std::cout << "Analytic differentiation: " << std::endl;
    for (auto i = 0; i < dataset.cols(); ++i)
    {
        ceres::CostFunction *cost_function = new AnalyticEllipseCostFunction(dataset.col(i));
        problem.AddResidualBlock(cost_function, nullptr, params);
    }
}

// Perform a gradient check
int check_gradients(const Eigen::Matrix2Xd &dataset, double *params, double tolerance)
{
    // First we create an instance of the cost function we want to check
    Eigen::Vector2d observed_point = dataset.col(0);
    auto cost_function = std::make_shared<AnalyticEllipseCostFunction>(observed_point);

    const double *parameters[] = {params};

    // We can use this object to customise the checking process
    ceres::NumericDiffOptions numeric_diff_options;
    ceres::GradientChecker gradient_checker(cost_function.get(), nullptr, numeric_diff_options);

    // We perform a probe. If unsuccessful, we can view the erroneous
    // gradients by writing the error log to the console
    ceres::GradientChecker::ProbeResults results;
    if (!gradient_checker.Probe(parameters, tolerance, &results))
    {
        std::cerr << "An error has occurred:\n"
                  << results.error_log;
        return EXIT_FAILURE;
    }

    std::cout << "Gradients correct!" << std::endl;
    return EXIT_SUCCESS;
}

std::string to_string(const double *params)
{
    std::stringstream buff;
    buff << "(h=" << params[0] << ", k=" << params[1] << ", a=" << params[2] << ", b=" << params[3] << ")";
    return buff.str();
}

DEFINE_string(mode, "numeric", "Mode for the program (one of 'numeric', 'autodiff', 'analytic', 'check_grad')");
DEFINE_int32(num_observations, 100, "Number of observations");
DEFINE_bool(verbose, false, "Output a verbose summary of the optimization");
DEFINE_bool(dump_data, false, "Whether to dump the data to a csv");

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    gflags::SetUsageMessage("Ceres Example");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    double initial_params[4] = {0.1, 0.3, 0.9, 1.2};
    double target_params[4] = {-0.3, 0.5, 4.3, 2.1};
    double params[4];
    Eigen::Matrix2Xd dataset = create_dataset(FLAGS_num_observations, target_params);

    if (FLAGS_dump_data)
    {
        std::ofstream output(FLAGS_mode + "_data.csv");
        output << "x,y" << std::endl;
        for (auto i = 0; i < dataset.cols(); ++i)
        {
            output << dataset(0, i) << "," << dataset(1, i) << std::endl;
        }
    }

    ceres::Problem problem;

    std::copy(initial_params, initial_params + 4, params);

    if ("autodiff" == FLAGS_mode)
    {
        setup_autodiff(problem, dataset, params);
    }
    else if ("numeric" == FLAGS_mode)
    {
        setup_numeric(problem, dataset, params);
    }
    else if ("analytic" == FLAGS_mode)
    {
        setup_analytic(problem, dataset, params);
    }
    else if ("check_grad" == FLAGS_mode)
    {
        return check_gradients(dataset, params, 1e-9);
    }
    else
    {
        std::cout << "Unrecognized mode: " << FLAGS_mode << std::endl;
        return 1;
    }

    // we can set upper and lower bounds for all parameters
    problem.SetParameterLowerBound(params, 2, 1e-5);
    problem.SetParameterLowerBound(params, 3, 1e-5);

    // The solver has a wide variety customization options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 8;

    // Here we add our own custom callback for logging to a file
    options.update_state_every_iteration = true;
    std::shared_ptr<CSVCallback> callback = std::make_shared<CSVCallback>(FLAGS_mode + "_fit.csv", params, FLAGS_num_observations);
    options.callbacks.push_back(callback.get());

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (FLAGS_verbose)
    {
        std::cout << summary.FullReport() << std::endl;
    }
    else
    {
        std::cout << summary.BriefReport() << std::endl;
    }

    std::cout << "Initial: " << to_string(initial_params) << std::endl
              << "Final: " << to_string(params) << std::endl
              << "Target: " << to_string(target_params) << std::endl;

    return 0;
}