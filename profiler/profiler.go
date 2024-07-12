package profiler

type ProfilerGroup interface {
	Start(label string) ProfilerGroup
	End()
}
