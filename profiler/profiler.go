// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package profiler

type ProfilerGroup interface {
	Start(label string) ProfilerGroup
	End()
}
