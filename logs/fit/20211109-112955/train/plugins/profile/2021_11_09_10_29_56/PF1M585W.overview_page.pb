?($	h??h??yv?=?^????ܵ?|??!??????$	?xy??@gSc?0?@??/?2???!M????*@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$鷯???/?$????AQ?|a??Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X?2ı.??ё\?C???A|a2U0*??YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ͪ??V??;?O??n??A?/L?
F??Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ܵ???Gr?????A??Q????Y?p=
ף??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?_vO????ׁsF??Av??????Y8gDio??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????H?}8g??AK?46??YGx$(??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?x?&1???ʡE????A??_?L??Y??ڊ?e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&_?Q???鷯????A=?U?????Y?ܵ?|У?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??6?[????K7?A??A?4?8EG??Y?R?!?u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	2U0*???m???????A???QI???YS?!?uq??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?h o???vq?-??A$(~????Y??镲??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?1??%???
h"lxz??AbX9????Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&w-!?l??Ș?????A`vOj??Yj?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ڬ?\m?????ʡE??Aۊ?e????Yj?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??MbX????W?2???A9??m4???Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?W?2??y?&1???A46<???Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&jM??St???w??#???A???H.??Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?? ?????????A??9#J{??YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&]?Fx??????o??A?ݓ??Z??Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O??e???????<,??AaTR'????Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ܵ?|??????Mb??A2w-!???YL7?A`???*	???????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatJ+???!R?g??C@)???ZӼ??1?#P???A@:Preprocessing2F
Iterator::ModelO??e???!??+MX?A@)?7??d???1?	+6?W6@:Preprocessing2U
Iterator::Model::ParallelMapV2?????̼?!!Y?B*@)?????̼?1!Y?B*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?A?f???!2j?S'P@)w??/ݴ?1??k?"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicez6?>W??!??uI??@)z6?>W??1??uI??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??|?5^??!~F??Q?'@)a??+e??1p???? @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap}гY????!O??Z?0@)?f??j+??1A???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*^K?=???!:?{.?@)^K?=???1:?{.?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9w???G#@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?S9A???M??@W??/?$????!H?}8g??	!       "	!       *	!       2$	!Ʒ??????8?????2w-!???!??9#J{??:	!       B	!       J$	BU???x??k?Ǖ`???ZӼ???!Gx$(??R	!       Z$	BU???x??k?Ǖ`???ZӼ???!Gx$(??JCPU_ONLYYw???G#@b Y      Y@q?F?nC?B@"?
both?Your program is POTENTIALLY input-bound because 49.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?37.8302% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 