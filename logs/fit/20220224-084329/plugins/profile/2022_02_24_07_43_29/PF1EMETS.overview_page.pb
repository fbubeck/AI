?&$	?"?F?????/?CGT???bG?P???!-$`tys??$	???S??@Ϻ???????4g??@!?ڰ???@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0-$`tys??a?????AV?j-?B??Yڌ?U???rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0n???8??wj.7???A+???????Y?A?Ѫ???rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?:?p?G????????AձJ??^??Y?!??gx??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?UfJ?o??Z?h9?C??AQ????+??Y?|zlˀ??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0W???x????^???A???\????Yޯ|?y??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?pt??????	1?Tm??A+L?k???Y?`?>??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??????RF\ ???AY?????Y????g??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	??I?2??1?q?P??A?H/j????Y????G??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
???͎T??p?71$'??A??m???Y??????rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0!?> ?M??Ǽ?8d??Aj??%!??Y??ʡE???rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0=D?;???-?i??&??AB'?????Y?P?n???rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0o?u?????h?
???Al??+??Y?uS?k%??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?mm?y???e?X???A?׺????Y3R臭???rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Ւ?r0???CSv?A]??A<?\?g??Y??q????rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?;?_?E???	j????A??H??_??Y?Yh?4??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?bG?P????'????A)[$?F??Y??#?????rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0j1x??????rh??|??A???????Y??F>?x??rtrain 97*	?rh??/?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??Z	?%??!??:I?B@)TR'?????1w???ŝA@:Preprocessing2E
Iterator::Root?Sȕz??!?P?gxD@)2??n??1?b?f?:@:Preprocessing2T
Iterator::Root::ParallelMapV2???9??!????F,@)???9??1????F,@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipX zR&5??!??G??M@)ŭ??ڧ?1C????Z@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?΢w*???!??`?u@)?΢w*???1??`?u@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenatelC?8??!i?> '@)p?󧍚?1??V??@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapRE?*k???!??xG??,@)??ȯ??1?輘@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*:?6U??!?f#?d???):?6U??1?f#?d???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9k??)?@I?>?bM?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	f????l??V?a]???'????!a?????	!       "	!       *	!       2$	?x???J????=?v֐??H/j????!B'?????:	!       B	!       J$	?????????u?_Ce??`?>??!??q????R	!       Z$	?????????u?_Ce??`?>??!??q????b	!       JCPU_ONLYYk??)?@b q?>?bM?W@Y      Y@qr???2U@"?	
both?Your program is POTENTIALLY input-bound because 48.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?84.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 