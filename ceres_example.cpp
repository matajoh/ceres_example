#include <sstream>
#include <fstream>
#include <iostream>
#include <random>
#include <memory>
#include <array>

#include <Eigen/Core>
#include <gflags/gflags.h>
#include <ceres/ceres.h>


template<typename T>
class Ellipse
{
public:
    Ellipse() : Ellipse(0, 0, 1, 1)
    {
    }

    explicit Ellipse(T h, T k, T a, T b)
    {
        m_data[0] = h;
        m_data[1] = k;
        m_data[2] = a;
        m_data[3] = b;
    }

    explicit Ellipse(const T* data)
    {
        std::copy(data, data + 4, m_data.begin());
    }

    explicit Ellipse(const Ellipse& other) : Ellipse(other.params())
    {        
    }

    Ellipse& operator= (const Ellipse &other)
    {
        std::copy(other.m_data.begin(), other.m_data.end(), m_data.begin());
        return *this;
    }

    const T* params() const
    {
        return m_data.data();
    }

    T* params()
    {
        return m_data.data();
    }

    T h() const
    {
        return m_data[0];
    } 

    T k() const
    {
        return m_data[1];
    }

    T a() const
    {
        return m_data[2];
    }

    T b() const
    {
        return m_data[3];
    }

    void set_bounds(ceres::Problem& problem)
    {
        problem.SetParameterLowerBound(params(), 2, 1e-5);
        problem.SetParameterLowerBound(params(), 3, 1e-5);
    }

    std::string to_string()
    {
        std::stringstream buff;
        buff << "(h=" << h() << ", k=" << k() << ", a=" << a() << ", b=" << b() << ")";
        return buff.str();
    }

private:
    std::array<T, 4> m_data;
};

/** Sample functor which simply computes the residual.
 *  Used for numeric differentiation.
 */
struct NumericEllipseCostFunctor
{
    NumericEllipseCostFunctor(const Eigen::Vector2d &observed_point) : observed_point(observed_point) {}
    bool operator()(const double *const parameters, double *residuals) const
    {
        Ellipse<double> ellipse(parameters);

        // compute the cost
        const double dx = observed_point.x() - ellipse.h();
        const double dy = observed_point.y() - ellipse.k();
        const double a2 = ellipse.a() * ellipse.a();
        const double b2 = ellipse.b() * ellipse.b();
        residuals[0] = (dx * dx) / a2 + (dy * dy) / b2 - 1;
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
        Ellipse<T> ellipse(parameters);

        T dx = T(observed_point.x()) - ellipse.h();
        T dy = T(observed_point.y()) - ellipse.k();
        T a2 = ellipse.a() * ellipse.a();
        T b2 = ellipse.b() * ellipse.b();
        residuals[0] = (dx * dx) / a2 + (dy * dy) / b2 - T(1.0);
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
        Ellipse<double> ellipse(parameters[0]);

        // We can re-use all of these later
        const double dx = observed_point.x() - ellipse.h();
        const double dy = observed_point.y() - ellipse.k();
        const double dx2 = dx * dx;
        const double dy2 = dy * dy;
        const double a2 = ellipse.a() * ellipse.a();
        const double b2 = ellipse.b() * ellipse.b();
        residuals[0] = dx2 / a2 + dy2 / b2 - 1;

        // will be null if only evaluating residuals
        if (jacobians != nullptr)
        {
            // if some parameters are being held constant,
            // then individual Jacobian matrices will also be null
            // to avoid unneeded computation
            if (jacobians[0] != nullptr)
            {
                using jacobian_t = Eigen::Matrix<double, 1, 4, Eigen::RowMajor>;
                Eigen::Map<jacobian_t> jac(jacobians[0]);
                jac(0, 0) = (-2 * dx) / a2;
                jac(0, 1) = (-2 * dy) / b2;
                jac(0, 2) = (-2 * dx2) / (a2 * ellipse.a());
                jac(0, 3) = (-2 * dy2) / (b2 * ellipse.b());
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
        Ellipse<double> ellipse(m_params);
        m_output << summary.cost / m_num_observations << "," << ellipse.h() << "," << ellipse.k() << "," << ellipse.a() << "," << ellipse.b() << std::endl;
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
                                const Ellipse<double> &ellipse,
                                double start_angle = -0.5,
                                double end_angle = 2.0,
                                double noise_sigma = 0.05)
{
    Eigen::RowVectorXd angles = Eigen::RowVectorXd::LinSpaced(num_observations, start_angle, end_angle);
    Eigen::Matrix2Xd data(2, num_observations);
    data.row(0) = (ellipse.a() * angles.array().cos()) + ellipse.h();
    data.row(1) = (ellipse.b() * angles.array().sin()) + ellipse.k();
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
    using cost_t = ceres::NumericDiffCostFunction<NumericEllipseCostFunctor,
                                                  ceres::CENTRAL, // method to use
                                                  1,              // # residuals
                                                  4>;             // # params
    for (auto i = 0; i < dataset.cols(); ++i)
    {
        ceres::CostFunction *cost_function =
            new cost_t(new NumericEllipseCostFunctor(dataset.col(i)));
        problem.AddResidualBlock(cost_function, nullptr, params);
    }
}

// Setup the problem using automatic differentiation.
void setup_autodiff(ceres::Problem &problem, const Eigen::Matrix2Xd &dataset, double *params)
{
    std::cout << "Automatic differentiation: " << std::endl;
    using cost_t = ceres::AutoDiffCostFunction<AutoEllipseCostFunctor,
                                               1,   // # residuals
                                               4>;  // # params
    for (auto i = 0; i < dataset.cols(); ++i)
    {
        ceres::CostFunction *cost_function =
            new cost_t(new AutoEllipseCostFunctor(dataset.col(i)));
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

DEFINE_string(mode, "numeric", "Mode for the program (one of 'numeric', 'autodiff', 'analytic', 'check_grad')");
DEFINE_int32(num_observations, 100, "Number of observations");
DEFINE_bool(verbose, false, "Output a verbose summary of the optimization");
DEFINE_bool(dump_data, false, "Whether to dump the data to a csv");

int main(int argc, char **argv)
{
    gflags::SetUsageMessage("Ceres Example");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    Ellipse<double> initial(0.1, 0.3, 0.9, 1.2);
    Ellipse<double> target(-0.3, 0.5, 4.3, 2.1);
    Ellipse<double> ellipse;
    Eigen::Matrix2Xd dataset = create_dataset(FLAGS_num_observations, target);

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
    ellipse = initial;

    if ("autodiff" == FLAGS_mode)
    {
        setup_autodiff(problem, dataset, ellipse.params());
    }
    else if ("numeric" == FLAGS_mode)
    {
        setup_numeric(problem, dataset, ellipse.params());
    }
    else if ("analytic" == FLAGS_mode)
    {
        setup_analytic(problem, dataset, ellipse.params());
    }
    else if ("check_grad" == FLAGS_mode)
    {
        return check_gradients(dataset, ellipse.params(), 1e-9);
    }
    else
    {
        std::cout << "Unrecognized mode: " << FLAGS_mode << std::endl;
        return 1;
    }

    // we can set upper and lower bounds for all parameters
    ellipse.set_bounds(problem);

    // The solver has a wide variety customization options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 8;

    // Here we add our own custom callback for logging to a file
    options.update_state_every_iteration = true;
    std::shared_ptr<CSVCallback> callback = std::make_shared<CSVCallback>(FLAGS_mode + "_fit.csv", ellipse.params(), FLAGS_num_observations);
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

    std::cout << "Initial: " << initial.to_string() << std::endl
              << "Final: " << ellipse.to_string() << std::endl
              << "Target: " << target.to_string() << std::endl;

    return 0;
}