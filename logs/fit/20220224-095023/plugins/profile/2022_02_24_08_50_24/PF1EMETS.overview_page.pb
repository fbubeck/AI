?&$	X?G????`?K9c???{ds?<??!??-W?6??$	bmK???@??c;???p?Ā@!Q͝e.?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0\?M4???崧?????A??S:X???Y???,??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??-W?6????gy???A??M~?N??Y??ajK??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0[닄????@Pn?????A?À%W??Y}Yک?ܐ?rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?{ds?<??"p$?`S??A^??-???Y?^EF$??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0`?|x????????U???A?}?k?,??YӥI*S??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?x]?`??ԀAҧU??A?	/????Y?m?2d??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????????A?w?????A?????B??Y???/?^??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?£?#????f?v???A?IEc????Y2??8*7??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
t??????eo)???A?(]?????Y,,?????rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?7?ܘ???)?1k???AKO?\??Y;S???.??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?!7?x????Ң>???A?@.q????Y[@h=|???rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?<HO?C????c?M*??A???>e??Yv7Ou?͐?rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?D?<?????6????A3?&c`??Y??_ ???rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0]?&?R??k?K????A????u6??Yb?1?縷?rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ı.n?????w.????Ad\qqTn??Y?;Ū??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?Pk?w???~??k	???ARD?U????Y	m9?⪒?rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?Q????Q???`???A~??7??Y8?Jw?ِ?rtrain 97*	??S???@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatc&Q/?4??!=?e9??A@)N^????1?IЦ?v@@:Preprocessing2T
Iterator::Root::ParallelMapV2?'-\Va??!?=T@??1@)?'-\Va??1?=T@??1@:Preprocessing2E
Iterator::Root?/L?
F??!??<+??A@)?7k??*??1] %??1@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip???:q9??!??a??%P@)?jׄ?ƨ?1???`?"@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice<?(A???!?B??O?!@)<?(A???1?B??O?!@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate!:???!?C!??.@)??ډ???1??.>7@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapn?HJz??!???X?3@)3??????1jD?c?=@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*S?1?#??!s?T)?@)S?1?#??1s?T)?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9N~???@I??2{X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$		?b!??????=h?,??A?w?????!??gy???	!       "	!       *	!       2$	?c?/va????6??~??^??-???!?IEc????:	!       B	!       J$	??&Zn??z? ?l{o?b?1?縷?!???/?^??R	!       Z$	??&Zn??z? ?l{o?b?1?縷?!???/?^??b	!       JCPU_ONLYYN~???@b q??2{X@Y      Y@qB????-U@"?	
both?Your program is POTENTIALLY input-bound because 45.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?84.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 