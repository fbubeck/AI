?&$	??jBU???1ևL?I???CV???!U?M?MS??$	??=\,8@?/???&?????0@!???2?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??1?????d??Ab??????Y?]i????rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???G6??????,'???Ain??K??Y??X??+??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0M?O???Փ?Gߤ??A????Z??Y2 Tq???rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0R???<H??Ȳ`⏢??A~??g??Y"lxz?,??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0!??q4???
?]?V??AdZ???Z??Y?el?f??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?ʠ??D??]m???{??A?z????Yq?-???rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?+d?*???5?e???AmFA????Y??"?ng??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	????I??????t???A??̒ 5??Y???<????rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
?CV???!ɬ??v??AE? ???Y??*3????rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0j?0
????5_%???A?y?):???Y*?~????rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0/O??R??"ߥ?%???A;?Ԗ:???Yo?;2V???rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??C????3?68???A)??q??Y2t젒?rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?2d????cAaP????A??N?0???YcFx{??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???:s????U+~???A??%?"??Y@a??+??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?I?5?????\?E??Ap?DIH???Y??R??q??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0U?M?MS??,??? ??A:?S?????YH4?"1??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Z??ڊ}??????2??A͓k
dv??YuZ?A????rtrain 97*	???K7	?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat
J?ʽ???!???}?A@)rQ-"????1`?|??@@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?pX?Q??!+Q:6?Q@)Va3?ٺ?1?????(/@:Preprocessing2T
Iterator::Root::ParallelMapV2R?r????!ݐ ?ݻ-@)R?r????1ݐ ?ݻ-@:Preprocessing2E
Iterator::Roott?????!V??'?=@)??ť*m??1̥V?p?-@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?::?Fv??!?????!@)?::?Fv??1?????!@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?*??<???!?????+@)???2n??1)(5??c@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??ŉ?v??!uI??3@)xF[?D???1????@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??E????!?U????@)??E????1?U????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 47.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9o?+^?@Iu?M0X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	2cf?????H?F????"ߥ?%???!,??? ??	!       "	!       *	!       2$	?~{=?q??}s???Y??E? ???!͓k
dv??:	!       B	!       J$	??@???????0?6o???"?ng??!??R??q??R	!       Z$	??@???????0?6o???"?ng??!??R??q??b	!       JCPU_ONLYYo?+^?@b qu?M0X@Y      Y@qA?V??O@"?	
both?Your program is POTENTIALLY input-bound because 47.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?63.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 